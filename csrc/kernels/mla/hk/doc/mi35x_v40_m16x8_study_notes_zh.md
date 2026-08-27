# MLA V4.0 decode kernel 学习笔记（mi35x m16x8 fp8bf16, gen1）

> 对象文件：`csrc/kernels/mla/hk/mi35x_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1.cuh`
> 配套精读材料：`doc/hk_mla_v40_gen1_spec.md`（2400 行设计 spec，本笔记按章节交叉引用为 "spec Ch.x"）
>
> 本笔记是「导游图」：目标是第一遍就能把 kernel 的骨架读通，每节末尾指出 spec
> 里对应的深挖章节。可以边读边在文末「待深挖问题」里追加问题，我们一起迭代。

---

## 1. 名字解码：这个 kernel 是什么

`mi35x_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1`，逐段拆开：


| 片段                | 含义                                                                            |
| ----------------- | ----------------------------------------------------------------------------- |
| `mi35x`           | 目标架构 gfx950（MI350X/MI355X），非 gfx950 时编译成 `assert(false)` 空壳（L1314）            |
| `v40`             | V4.0 数据布局：NoPE 与 RoPE **分 buffer** 存放；NoPE 缩为 448 维 fp8（V3.2 是 512 维同 buffer） |
| `fwd_decode`      | MLA（Multi-head Latent Attention）decode 阶段前向，qseqlen 很短（1~8，MTP）               |
| `m16x8`           | 8 个 warp，每个 warp 负责 16 行 Q（kBlockM = 128 = 8×16）                              |
| `fp8bf16_fp8bf16` | Q 侧 NoPE=fp8 / RoPE=bf16，KV 侧 NoPE=fp8 / RoPE=bf16                            |
| `gen1`            | 该体系第一代实现（寄存器图/manager 均带 gen1 后缀）                                             |


一句话概括：**一个 persistent、split-KV、fp8 输入 bf16 计算的 MLA decode
attention kernel，手工钉死 VGPR 布局，LDS 双缓冲流水 KV，softmax 全程 log2 域，
8 个 warp 按 Lo/Hi 两种编译期角色做 MFMA/VALU 乒乓交错。**

MLA 的关键数学性质：K 和 V 是**同一份** latent 张量（kv_lora），
`S = Q·Kᵀ`（d_qk = 448 NoPE + 64 RoPE = 512），`O = P·V` 其中 V 就是 K 的全部
512 列（含 RoPE 尾巴——这是 V4 与 V3.2 的区别之一，见 `hk_mla_utils.cuh:133`）。
所以 LDS 里只需要一份 KV tile，QK 和 PV 都吃它。

→ spec Ch.1、Ch.2.1–2.3

---



## 2. 输入输出与数据布局



### 2.1 张量清单（host 侧检查见 L1420–1571）


| 张量                           | 形状                            | dtype | 备注                                                                 |
| ---------------------------- | ----------------------------- | ----- | ------------------------------------------------------------------ |
| `query`                      | [total_q, H, 512]             | fp8   | **packed NoPE**：448 fp8 + 14 B e8m0 scale + 50 B pad = 512 B/token |
| `query_rope`                 | [total_q, H, 64]              | bf16  | RoPE 原生布局                                                          |
| `kv_buffer`                  | [num_page, page_size, 1, 512] | fp8   | 同样的 packed NoPE，paged                                              |
| `kv_buffer_rope`             | [num_page, page_size, 1, 64]  | bf16  |                                                                    |
| `final_output`               | [total_q, H, 512]             | bf16  | 单 split 直接写                                                        |
| `split_output` / `split_lse` | [..., 512] / [...]            | fp32  | 多 split 时写中间结果 + LSE，由外部 reducer 合并                                |
| `attn_sink`                  | [H]（可选）                       | fp32  | attention sink logit，缺省为 -inf（等效关闭）                                |


packed NoPE 的 512 B：`448 fp8 | 14×e8m0 | 50 pad`。e8m0 是 1 字节的 2 的幂
scale，每 32 元素一个 slot，但真实量化 tile 是 64 元素（`scales[2i]==scales[2i+1]`
复制存放，kernel 每 64 列 strip 只读 1 字节）。→ `hk_mla_utils.cuh:145`

### 2.2 关键编译期常量（`HkMlaV40DecodeFwdTraits`，`hk_mla_utils.cuh:131`）

```
kQkNopeHeadDim=448   kQkRopeHeadDim=64   kQkHeadDim = kVoHeadDim = 512
kBlockM=128  kBlockN=64  kBlockK=32  kTileM=16
kNumWarps=8 (512 线程, wave64)   kOccupancy=1
kPageSize ∈ {1, 64}（只实例化这两个，L1604）
kRescaleThreshold=8.0（online-softmax 免 rescale 阈值，见 §6.4）
```

---



## 3. 执行模型：persistent kernel + work scheduler

- grid = CU 数（L1378），每个 block 常驻一个 CU，动态 LDS 要满 160 KB（`maxSharedMemoryPerMultiProcessor / kOccupancy`）。
- 工作划分由外部 planner（metadata kernel）生成，本 kernel 只消费：
  - `work_indptr[blockIdx.x .. +1]`：本 worker 的 work item 区间（L69）；
  - `work_info_set`：每个 work item 7 个 dword（L248–262）：


| dword | 字段                  | 含义                                                                                      |
| ----- | ------------------- | --------------------------------------------------------------------------------------- |
| 0     | `batch_idx`         | 批次索引                                                                                    |
| 1     | `partial_qo_loc`    | `<0` → 本 item 是完整 batch，直接写 `final_output`；`≥0` → split 槽位，写 `split_output`+LSE         |
| 2,3   | `qo_start/end`      | Q 行范围                                                                                   |
| 4,5   | `kv_start/end_page` | KV page 范围（×kPageSize 转 token，尾页用 `kv_last_page_lens` 截断）                               |
| 6     | `kv_offset`         | 本 split 结束点距 batch 真实结尾的 token 数；**== 0 当且仅当是该 batch 的最后一个 split**（sink 折叠只在这时做，见 §6.6） |


- **causal / MTP**：qseqlen = kBlockM / H ∈ {1,2,4,8}。warp 的 16 行属于同一个
q 位置组，`qpos_off_from_last`（L183）算出该 warp 的 q 相对最后一个 token 的
偏移，得到 per-warp 的 `kv_end_eff`——靠后的 q 位置多看几个 KV token，
靠前的少看，尾部差异 < kBlockN，所以最多多出 1 个「空转 iter」（§7）。

→ spec Ch.2.4、Ch.12.7–12.8

---



## 4. Block 内分工：8 个 warp 各干什么

**计算侧人人平等**：每个 warp 独立算「自己的 16 行 Q × 整个 64 行 KV tile」的
QK/softmax/PV，互不依赖（M 维切分）。

**KV 搬运侧按 band 分工**（V2 band remap，L356–357）：一个 64 行 × 512 列的
KV tile 拆成 8 块 16 行 × 256 列，每 warp 搬一块：


| warp | 行（band = w&3） | 列（tile） | 角色                                        |
| ---- | ------------- | ------- | ----------------------------------------- |
| 0    | 0–15          | 0–255   | `LoNoPEWarp`（纯 NoPE）                      |
| 1    | 16–31         | 0–255   | Lo                                        |
| 2    | 32–47         | 0–255   | Lo（写入 sub-tile B 区，LDS +32 KB）            |
| 3    | 48–63         | 0–255   | Lo                                        |
| 4    | 0–15          | 256–511 | `HiRoPEWarp`（NoPE 256–447 + RoPE 448–511） |
| 5    | 16–31         | 256–511 | Hi                                        |
| 6    | 32–47         | 256–511 | Hi                                        |
| 7    | 48–63         | 256–511 | Hi                                        |


warp 类型在 `__global__` 入口一次性分流（L1302），Lo/Hi 各自编译成独立函数体
（`kWarpType` 是模板参数）。Lo/Hi 的差异不止搬运列不同：


| 差异点            | Lo（0–3）                                       | Hi（4–7）                                           |
| -------------- | --------------------------------------------- | ------------------------------------------------- |
| 搬的列            | 4 个 fp8 strip                                 | 3 个 fp8 strip + 1 个 bf16 RoPE strip               |
| RoPE           | 无                                             | strip 3 用 `buffer_load_lds` DMA 直达 LDS（bf16 无需转换） |
| strip 3 (NoPE) | 存 staging，**推迟到下一个 iter 开头才 cvt+store**（§6.2） | 不适用                                               |
| PV 时机          | **call 末尾**（`kPvAtEnd=true`）                  | **推迟一个 tile**，在下一个 call 开头做上一 tile 的 PV（`kHasPv`） |
| softmax ALU    | packed（`v_pk_*`）                              | de-packed                                         |
| epilogue 归一化   | de-packed                                     | packed                                            |


最后三行是刻意「错相位」：occupancy=1 时每个 SIMD 上住着一对 Lo+Hi wave，
MFMA/LDS 管线是共享的——一个 wave 在跑 PV 时另一个只能发 VALU/SALU/vmem。
Lo 在 call 尾做 PV、Hi 在 call 头做（上一 tile 的）PV，两者错开半个 tile，
保证任意时刻大约一个在 MFMA、一个在 softmax/搬运，把 CU 填满。packed ALU
的使用也按相位互补分配。**这是整个 kernel 最核心的设计**。

⚠️ 注意：kernel 源码里 deferred-PV 相关的几处注释把 Lo/Hi 写反了
（如 L543 "Lo deferred PV"、L869 "Lo drain"，代码守卫其实是 `!kPvAtEnd` = Hi）。
以代码和 spec Ch.3.5/10.5.1 为准：**Lo = PV at end，Hi = deferred PV**。

→ spec Ch.3.4–3.5、Ch.10.5.1

---



## 5. 存储层次：手工管理的 VGPR 与 LDS



### 5.1 钉死的 VGPR 布局（`HkMlaV40Regs`，common 头 L24）

`__global__` 上挂 `__attribute__((amdgpu_num_vgpr(36)))`（L1292）：编译器只准用
v0–v35 做 scratch；v36–v255 全部由 inline asm 按十六进制编号手工使用，
`hkdart::clobber<>()` 声明占用，`hk::art<...>`（auto register tile）给这些裸
寄存器段套上「类型 + 布局」的静态视图。

每 lane 256 个 VGPR 的分配（kBlockN=64 版）：

```
v255┐
    │ oaccu       128 个 fp32 = 16 行 × 512 列输出累加器 / 64 lane
v128┘
v127┐
    │ q_vgpr      64 个 = 整个 Q 16×512 bf16（col-tile 0–13 NoPE, 14–15 RoPE）
v 64┘              整个主循环只读不写 —— QK 的 A 操作数零 LDS 读
v 63┐ p_comp      16 个 fp32 = 本 tile 的 64 列 QK 分数（A: v48–55, B: v56–63）
    │   ├ p_mfma  v48–55：softmax 后原地 pack 成 bf16 的 P（overlay）
v 48┘   └ pv_v_0  v60–63：PV 期间借 p_comp 高 4 个当 V 槽位（QK 期间没人用）
v 47┐ k_2 / pv_v_3  QK 的 K ds_read 三槽环形缓冲；PV 期间同一批寄存器
v 40┤ k_1 / pv_v_2  改当 V 的 4 槽环形缓冲（k0=v36, k1=v40, k2=v44）
v 36┘ k_0 / pv_v_1
v 35┐
    │ 编译器 scratch（地址、标量搬运等）
v  0┘
```

我们来算一下预先分配的这些寄存器都是做什么的.

**先把 shape 流写出来**（一个 work item 的逻辑视角）：

```
Q[128, 512] @ Kᵀ[512, T] → S[128, T] → P = softmax(S)
P[128, T] @ V[T, 512] → O[128, 512]
```

128 = H × qseqlen（M 维）；**T 不是 128**，它是本 work item 的 KV 长度
（任意长的上下文），被切成 kBlockN=64 行一个的 tile 逐个消化。「逐个消化」
正是下面第 2、3 条的答案。

1. **oaccu：`128 fp32/lane` → `[16, 512]/warp` → `[128, 512]/cta`，
   是 PV 的累加结果（分子）。**

   - warp i 拿 O 的行 [16i : 16(i+1))，全部 512 列——「×8」发生在 M 维。
   - lane 级映射：矩阵乘是 v_mfma_f32_**16x16x32**_bf16，D tile 16×16。
     标准 CDNA 布局是 lane = 16g+m 持有 D[4g:4g+4, m]（即 "[:4, 0]"）；
     但本 kernel QK/PV 都**故意 swap 了 A/B**（K/V 当 A 操作数），所以在
     [q行, 列] 坐标里正好是转置版：**lane l 持有 q 行 = l%16、
     列 = 16t + 4·(l/16) + (0..3)**，t = 第几个 16×16 tile。
   - 对 oaccu：t = 0..31（512/16 个 tile），每 lane 32×4 = 128 个 fp32，
     列号从 4·(l/16) 起、步长 16，直到 512。同一 q 行由 l%16 相同的
     4 个 lane 分摊（4 × 128 = 512 列）。
   - 代码锚点：epilogue LSE 注释 "lanes 0..15 own the M-rows"（L944）
     ⇒ q行 = lane%16；softmax 的 `col_0_idx = lane>>4` ⇒ 列组 = lane/16；
     PV 的 D 循环按「(oaccu_a, oaccu_b) = 32 列 = 8 VGPR」推进
     （`oaccu_base = k_o_begin + col_idx*8`），16 D-iter × 8 = 128 ✓。

2. **O_running（= oaccu 里存的东西）是什么**：online softmax
   （flash-attention）的「运行中的未归一化输出」。逐 tile 维护三个量——
   行最大 `m`、行 exp 和 `l`（代码里的 `row_sum_e`）、输出累加 `O_acc`：

   ```
   m_new = max(m, rowmax(S_tile))
   r     = exp(m − m_new)            # rescale；本 kernel 常被阈值 8 跳过
   O_acc = O_acc · r + exp(S_tile − m_new) · V_tile
   l     = l · r + rowsum(exp(S_tile − m_new))
   最后一个 tile 之后：O = O_acc / l
   ```

   oaccu 一直存分子 `O_acc`；分母 `l` 是每 lane 一个标量（`row_sum_e`），
   epilogue 才相除（§6.6）。

3. **QK 的结果存在哪？** 需要寄存器，但只要 **16 个**——就是图里的
   `p_comp`（v48–63）。逻辑上的 S 是 [16, T]/warp，永远**不整块物化**；
   每个 iter 只存在当前 tile 的 S_tile [16 q行 × 64 KV列] =
   1024 fp32 ÷ 64 lane = 16 VGPR。生命周期一条线：
   QK mfma 写入 → softmax **原地**变 `exp(S−m)` → pack 成 bf16 挤进低半段
   `p_mfma`（v48–55 overlay，体积减半）→ PV 当 B 操作数吃掉 →
   下个 tile 复用这 16 个寄存器。这就是 online softmax 省寄存器的全部意义：
   S 是瞬态、tile 大小的中间量，只有 O 和 (m, l) 需要跨 tile 存活。

- `q_vgpr = 64`：16 × 512 bf16 = 8192 × 2 B ÷ 64 lane ÷ 4 B = 64。
- `p_comp = 16`：64（KV 列）× 16（Q 行）fp32 ÷ 64 = 16。
- 常见误算：`128 × 8 = 32 × 32`——「×8」其实发生在 **M 维**：8 个 warp
各有一份互不重叠的 oaccu（不同的 16 行），合起来才是整个 block 的
128 行 × 512 列输出，没有 32×32 的块；32 真正出现的地方是
「每 warp 32 个 16×16 oaccu tile」。

设计要点：寄存器**分时复用**——同一物理寄存器在 QK 阶段叫 `k_`*、PV 阶段叫
`pv_v_*`；p_comp 的高段在 PV 时借给 V。整卡 0 spill。

→ spec Ch.5（含每个偏移的 rationale、scratch 预算审计）

### 5.2 LDS 布局（L185–237），合计正好 160 KB


| 偏移     | 大小    | 内容                                                                                           |
| ------ | ----- | -------------------------------------------------------------------------------------------- |
| 0      | 64 KB | `p_lds_kv_curr`：当前 KV pong（64 行 × 512 bf16；sub-tile A 前 32 KB、B 后 32 KB）                     |
| 64 KB  | 64 KB | `p_lds_kv_next`：下一 tile 的 pong；**epilogue 的 O bounce 覆盖在这里**（最后一个 iter 不再 swap，next pong 已死） |
| 128 KB | 16 KB | `p_lds_q`：QManager 的 staging / RoPE 终区（主循环内已死）                                               |
| 144 KB | 16 KB | `p_lds_kv_stage`：fp8 原始数据 staging（2 slot × 8 KB，每 warp 1 KB）                                 |


两个讲究的点（都有 static_assert 兜底，L223–232）：

1. **O bounce 为什么盖在 KV-next 而不是 Q 区**：Q 区每 warp 步幅与 OManager 不同，
  盖 Q 会和下一个 work item 的 load_q 产生跨 warp 竞争；盖 KV-next 则下一个
   使用者（下个 work 的 prologue）写的是 curr，天然错开。
2. **Q 区为什么放最后**：QManager Phase-1 会对 LDS 目的地址预减最多 384 B
  （`kLdsHeadPadBytes`），Q 前面垫着两个 KV pong 才有减的余量（m0 不下溢）。

→ spec Ch.4

---



## 6. 主循环：`mla_main` 一次迭代的时间线

外层：`for work_idx` → 读 work_info → 解析 KV 范围 → `load_q`（fp8→bf16、
**顺手把** `softmax_scale × log2e` **乘进 Q**，L371）→ prologue 预取 tile 0 →
按 dispatch ladder 连续调用 `mla_main`（一次一个 64 行 KV tile）。

一次 `mla_main`（L434–967）内部，按顺序：

```
Lo:  [strip3 消费(补齐 curr)]  [Phase A 预取 next]                 ─barrier─  QK  [Phase B+C cvt→next]        softmax  PV(本tile)  [epilogue?]  swap
Hi:                            [Phase A 预取 next]  PV(上一tile)   ─barrier─  QK  [RoPE DMA + Phase B+C]      softmax              [epilogue: 补PV+输出]  swap
                                                                   s_setprio(3)      s_setprio(2)                s_setprio(1)
```



### 6.1 Phase A：预取下一 tile（L483–541）

每 warp 把自己 16×256 band 的**下一 tile**数据发射出去（vmem 延迟藏在 barrier

- QK 底下）：strip 0、1 直接 `buffer_load_dwordx4` 落 VGPR（"carrier"），
strip 2（Lo 还有 strip 3）`buffer_load_lds` DMA 进 staging；每 strip 附带 1
字节 e8m0 scale。KV 行号（page 表查找）提前一个 tile 解析好放在
`row_kv_ld_next_next` 里滚动携带，且用 `asm("" : "+v")` 钉住，防止编译器把
buffer_load 重算进地址链引发全排水 waitcnt（L473 的注释是个经典教训）。



### 6.2 Lo 的 deferred strip-3（L499–520）

Lo 的 strip 3 在上一个 call 只 stage 不转换（scale 存在跨 iter 的 `s3_scale`
VGPR 里）；本 call 开头才 ds_read + cvt + 写进 **curr** pong ——把这份 VALU 工作
从繁忙的 Phase B 挪到 Hi 正在跑 PV 的窗口里，并让 Lo 自己的 PV 能更早开始
（详见 spec 10.5.1 第 3 点）。代价：curr tile 的 192–255 列直到本 call 开头
才就绪，恰好赶在 barrier + QK 之前。

### 6.3 QK GEMM（L574–672）

- 数学形式：`p_comp(N×M) = K_tile · Q_tileᵀ`，`mma_ABt(p_comp, k, q)`，
mfma 16×16×32 bf16，K 作 A 操作数、Q 作 B 操作数（分数天然转置存放，
方便后面按 N 列做 softmax mask）。
- 64 个子步 = 16 个 col-tile（512/32）× 4 个 row-group（A 两半 + B 两半），
4 个累加器 `p_comp_lo/hi/b_lo/b_hi` 各 16×16。
- K 从 curr pong `ds_read_b128` 读进 3 槽环形缓冲（k0/k1/k2），3 深软件流水：
预发 3 个读，之后「等到 S 是最老的未完成读（lgkmcnt=min(尾数,2)）→ mfma →
补发 S+3」。Q 全在 VGPR（`kQkGemmTiles=16`），lgkmcnt 只数 K 的 ds_read，
不需要任何精细调参。



### 6.4 Phase B+C 与 softmax（L678–838）

- **Phase B+C**：等 carrier/staging 落地（分级 vmcnt），fp8×e8m0→bf16 逐 16
字节转换（`cvt_kv_tile_step`），写进 **next** pong；Hi 同时发 RoPE DMA。
刻意排在 QK 之后，避免写 next 的 LDS 流量挤占 QK 读 curr。
- **softmax（全程 log2 域）**：Q 已预乘 `softmax_scale·log2e`，此处只做
OOB mask（按 `kv_end_eff` 两个 32 列 sub-tile 各查一次边界）→ 16 寄存器求
max → `warp_reduce`（同一 M 行组的 4 个 lane）→ online rescale 判定 →
`exp2` + 求和 → 原地 pack 成 bf16 `p_mfma`。
- **rescale-skip（kRescaleThreshold=8.0）**：新 max 比旧 max 高不到 8（自然对
数域）就**不更新 row_max、不做 rescale**——用 ballot 升格成 wave 一致决定
（`do_rescale`）。长上下文时 running max 早就平了，绝大多数 tile 走免乘法
路径，实测 ~3%。离 fp32 的 e⁸⁸ 溢出墙很远，精度实测 bit-exact。



### 6.5 PV GEMM（common 头 `hk_mla_v40_pv_gemm`，L169）

- 数学形式：`oaccuᵀ = Vᵀ · Pᵀ` 即 `mma_ABt(oaccu, v, p_mfma)`；V 直接从 KV
pong 用**转置 ds_read**（`ds_read_b64_tr_b16`）读出，无需单独存 V。
- 一次调用吃掉 A+B 两个 sub-tile：32 个 D-iter × 2 mfma；V 走 4 槽环形
（pv_v_0 + 借来的 k0/k1/k2），读槽 j%4、补槽 (j+3)%4，错开 ds_read 目的
寄存器和 mfma 源寄存器的 WAR。
- **rescale 折叠**：oaccu 乘 rescale 不整块做，而是拆成 `v_pk_mul_f32` 对，
塞在相邻 mfma 之间（只在 sub-tile A 趟内、且还有下一个 D-tile 时），
完全藏进 MFMA 阴影里。
- 首 iter：Lo 用 3-arg mfma 直接初始化 oaccu；Hi 因为 PV 推迟、首个 PV 走累加
路径，所以在首个 compute call 先 `hk::zero(oaccu)`（L452）。



### 6.6 Epilogue（L861–959）

只在全局最后一个 tile 的 call 里执行：

1. Hi 先补做本 tile 的 PV（deferred 的排水，L869）；
2. **attention sink 折叠**：`row_sum_e += exp2(sink·log2e − row_max)`，只在
  `OutputFinal` 或该 batch 的最后一个 split 做——保证 reducer 的全局分母里
   sink 恰好进一次，分子贡献为 0（L885）；
3. oaccu × `1/row_sum_e`（Lo/Hi 分别用 de-packed / packed 乘法）；
4. 经 O bounce LDS（盖在 next pong 上）做 sb8 逆置换 un-swizzle，写 bf16
  `final_output`，或 fp32 `split_output` + LSE（LSE 从 log2 域换回自然域，
   L955）。

→ spec Ch.8（KV 流水全景）、Ch.9（softmax）、Ch.10（PV）、Ch.11（epilogue/swizzle）

---



## 7. Dispatch ladder：把运行时分支变成模板实例（L969–1285）

`mla_main` 有 4 个编译期开关：


| 参数                   | 作用                                                     |
| -------------------- | ------------------------------------------------------ |
| `kIsFirstIter`       | 本 warp 第一个计算 iter：oaccu 初始化路径、softmax 无 rescale        |
| `kSkipCompute`       | causal 尾部空转 iter：只参与 barrier + 协作搬 KV，不算 QK/softmax/PV |
| `kEpilogueType`      | `None` / `OutputFinal` / `OutputSplit`                 |
| `kCheckBoundaryNext` | 预取下一 tile 是否要做边界检查（尾 tile 可能不满）                        |


外面的「梯子」按 `kv_len_eff` 的档位（0 / ≤1 tile / >1 tile）手工展开：首 iter、
中间 iter（**循环体内零分支**——把可能越界的最后一个中间 iter 剥到循环外，
实测 ~2-3%）、最后一个真实 iter（带或不带 epilogue）、可能的尾部 skip iter。

`MLA_SLIM_DISPATCH`（默认开，L19）：放弃「整除 kBlockN」的免检查快路径，
所有预取恒带边界检查——模板实例数减半换编译时间，代价每 iter 1 cmp+cmov。

→ spec Ch.12

---



## 8. 值得单独体会的工程手法清单

1. **编译期 warp 特化**：入口一次分流，Lo/Hi 各自成独立编译体，所有角色差异零运行时开销（L1302）。
2. `readfirstlane` **标量化**：所有 wave 一致的值（work_info、num_qheads 等）显式搬进 SGPR，帮编译器发 s_load/s_branch。
3. `sched_barrier(0)` **/** `s_setprio` **阶梯**：前者当调度器栅栏切分阶段，后者（3→2→1）告诉硬件仲裁器 QK > Phase B > softmax/PV 的相对优先级。
4. `asm("" : "+v"(x))` **钉寄存器**：阻止编译器 rematerialize 一条 buffer_load 到地址链上（否则引发 vmcnt(0) 全排水）——L469 注释记录了这个真实 bug。
5. **数据相关的 in-flight load 计数警告**（L322）：`resolve_row_kv_ld` 可能发 0 或 1 条 load，静态 vmcnt 数不了它，必须在相关 waitcnt 之后才 issue。
6. **数学折叠**：`softmax_scale·log2e` 一次乘进 Q；exp→exp2；LSE 最后一步才换回自然域。整个热路径没有一次「本可以提前做」的乘法。
7. **免 rescale 阈值 + ballot**：用数值分析换掉大多数 tile 的 oaccu 全扫乘法（§6.4）。
8. **LDS 复用三连**：O bounce 盖死 pong、Q staging 区被 RoPE 复用、pre-subtract 靠区域排序拿余量（§5.2）。
9. **推迟（defer）作为调度手段**：Hi 推迟整个 PV 一个 tile、Lo 推迟 strip-3 的 cvt——都不是省工作量，而是把工作挪进伙伴 wave 的 MFMA 窗口。

---



## 9. 已知的「读码陷阱」

- **stale 注释**：文件头 enum 注释（L21–25，"PV is always at call end"、"warps 5,7"）
和 deferred-PV 各块的 "Lo ..." 标注（L448、L543、L869）与代码极性相反，疑似从
m16x4 变体或旧版本遗留。判定依据：`kPvAtEnd = (kWarpType == LoNoPEWarp)`（L60）
  - spec Ch.3.5 表格。**Lo = PV at end；Hi = deferred PV**。
- L79–86 的 VGPR map 注释也是旧版（"chunks 0-4"、"v104..v111 q_lds scratch"）；
现行真相以 `HkMlaV40Regs`（common 头）+ spec Ch.5.2 为准（kQkGemmTiles=16，
Q 全量 512 列在 v64–127）。
- L187 "32 KB each" 是 kBlockN=32 时代的数字；kBlockN=64 下每个 pong 是 64 KB
（`get_lds_size_in_byte` = 64×512×2）。

---



## 10. 建议学习路径 & 待深挖问题

**路径**（由粗到细）：

1. 本笔记 §1–4 + spec Ch.1–3 —— 建立全景；
2. §6 对着源码把一次 `mla_main` 走通（建议先读 Lo 路径，再 diff Hi）；
3. spec Ch.5（VGPR）+ Ch.8（KV 流水）—— 两大机制精读；
4. spec Ch.6–7（QManager 与 sb8 置换）、Ch.9–11 —— 各阶段细节；
5. spec Ch.10.5.1 —— 回头理解为什么所有「推迟」都指向同一个乒乓设计。

**待深挖问题**（我们下次继续，可随时追加）：

- [ ] sb8 置换的闭式（spec 7.2）：为什么 Q 的 D 轴必须与 K 同置换？
- [ ] mfma 16×16 输出的 lane 布局与 `softmax` 里 `col_0_idx = lane>>4`、
  ```
  `warp_reduce(stop_stride=15)` 的对应关系（spec Ch.9）。
  ```
- [ ] `ds_read_b64_tr_b16` 转置读的硬件语义与 bank conflict 情况（spec 10.2、4.5）。
- [ ] QK 里 Lo 的 vmcnt=9 / Hi 的 vmcnt=6（L563–570）逐条 load 对账。
- [ ] m16x4 变体 diff：为什么它放弃乒乓（occupancy=2 跨 workgroup 不可控，spec 10.5.1 末段），V1 manager 差在哪。
- [ ] planner（metadata kernel）怎么切 split 与生成 work_info（`metadata/v1_2_device.cuh`）。
- [ ] 性能数字：thread trace 里 QK/PV/搬运各占多少（spec Ch.1 perf state）。
- [ ] HipKittens 的 art 实现：`hk::art` 怎么把 range 参数变成 inline asm 的
  ```
  `v[..:..]` 操作数、`mma_ABt` 到 `v_mfma_*` 的映射（见附录）。
  ```

---



## 附录：HipKittens 源码地图

kernel 里所有 `hk::` / `hkdart::` 原语来自 HipKittens（ThunderKittens 的 AMD
版）。aiter JIT 默认从 `3rdparty/HipKittens` 引用（`aiter/jit/core.py:482`，
env `HIP_KITTENS_DIR`），本仓库固定在 commit `d3cd9b31`。命名空间桥接在
`hk_mla_utils.cuh:13`：`hk = kittens`，`hkdart = hk::ducks::art`。

本 kernel 用到的构件 → HipKittens 内位置（均在 `include/cdna4/` 下）：


| kernel 里的写法                                | 定义处                                                  | 说明                                                   |
| ------------------------------------------ | ---------------------------------------------------- | ---------------------------------------------------- |
| `hk::art<T, M, N, layout, shape, ranges>`  | `types/register/art.cuh` + `art_base.cuh`            | assembly-mode 寄存器 tile：类型只是「视图」，物理寄存器由 ranges 模板参数指定 |
| `hkdart::range / type_list / split_many_t` | `types/register/art.cuh`（`ducks::art` 命名空间开头）        | 描述 VGPR 区段的类型级小语言                                    |
| `hkdart::clobber<ranges>()`                | `types/register/art.cuh`                             | 向编译器声明这些 VGPR 被占用                                    |
| `hk::rt_16x32_s / rt_16x16_s`              | `types/types.cuh:65` → `types/register/rt_shape.cuh` | mfma 基础 tile 形状（`_s` 是 shape 别名）                     |
| `hk::mma_ABt(d, a, b[, c])`                | `ops/warp/register/tile/assembly/mma.cuh`            | art 版 mfma 封装（普通 rt 版在同级 `tile/mma.cuh`）             |
| `hk::mul_vgpr / hk::zero`                  | `ops/warp/register/tile/assembly/maps.cuh`           | 对整段 pinned 区间的逐寄存器 map 操作                            |
| `hk::gl<T, ...>`                           | `types/global/gl.cuh`                                | global-memory tensor 描述符（traits 里的 `gl_q_nope` 等）    |
| `hk::st_fp8e4m3 / st_bf` + `st_16x16_s`    | `types/shared/st.cuh` / `st_shape.cuh`               | LDS tile 类型                                          |
| `hk::group<N>`                             | `ops/group/group.cuh`                                | warp 组协作原语                                           |
| `hk::bf16 / fp8e4m3 / u32x4`               | `common/base_types.cuh`                              | 基础标量/打包类型                                            |


阅读建议：先看 `art_base.cuh`（一个 art tile 怎么把 range 展开成 asm 操作数），
再对照 `assembly/mma.cuh` 看一条 `mma_ABt` 最终吐出的 `v_mfma_f32_16x16x32_bf16`
长什么样，之后回读 kernel 的 QK 循环会非常顺。

---

# 日志区（append-only，从这里往下当学习日志用）

## 2026-08-27 Q&A：QK 乘完为什么能「立刻」softmax、立刻 PV？

> 问题原文：s2r 读出来 q & k 然后相乘，然后直接就能 softmax 么？然后就能
> p & v 相乘了么？我记得好像得算完所有 dim 求出来 online softmax 的那个
> 最大值才行？为啥 p_tmp（p_comp）只用这么一点点就行？那么怎么求 row_max？

### 1. 先拆掉一个概念混淆：softmax 归约的轴是 T，不是 dim=512

S = Q·Kᵀ 有两类「维」，作用完全不同：

- **dim = 512（head dim）**：这是 GEMM 的收缩轴（K 轴）。它在 softmax
  之前就**必须也确实已经收完了**——QK 循环的 64 个子步（16 个 col-tile ×
  4 个 row-group，kernel L663）就是在把 512 维一段段（每段 32）累进
  p_comp。循环走完时，p_comp 里的每个数都是**完整**的
  `Σ_{d=0..511} q_d·k_d`，没有任何「没算完的 dim」。
- **T（KV token 数）**：这才是 softmax 求 max / 求和的轴。你记忆里的
  「要先算完所有才能求 max」指的就是这条轴——而它正是 online softmax
  要绕过去的东西。

所以流水线顺序「QK → softmax → PV」在**单个 tile 内部**没有任何数学妥协：
进入 softmax 时这 64 列分数每一个都是精确完整的。

### 2. tile 之间：确实不知道全局 max，但不需要知道

朴素 softmax 需要全局 `M = max_t s_t`。online softmax 的核心恒等式：

```
exp(s − M) = exp(s − m) · exp(m − M)      （对任意 m）
```

也就是说：用一个**暂时的、偏小的 max `m`** 先把 exp 算了、把 O 和分母都
累了，将来发现真正的 max 更大时，只要给**已累加的整体**补乘一个
`exp(m_old − m_new)`（就是 rescale `r`），结果和从头用全局 max 算**逐位
相等**。这是线性性：O_acc 和 l 都是「每项带同一个因子」的和，因子错了就
整体乘一次修正。

每行跨 tile 只需要 3 个存活量：`m`（running max）、`l`（row_sum_e，分母）、
`O_acc`（oaccu，分子）。每来一个 64 列 tile：

```
m_new = max(m, rowmax(S_tile))          # 只看本 tile 64 列 + 旧 m
r     = exp(m − m_new)
O_acc = O_acc·r + exp(S_tile − m_new)·V_tile
l     = l·r     + rowsum(exp(S_tile − m_new))
```

**数值例子**（一行、两 tile、每 tile 2 列）：分数 tile1 = [1, 3]，
tile2 = [5, 2]，全局 max 其实是 5。

- tile1 结束：m=3，l = e⁻² + 1，O_acc = e⁻²·v₁ + 1·v₂。
  （这里 v₂ 的权重 1 是「错的」，真值该是 e⁻²——先欠着。）
- tile2：local max = 5 > 3 → m_new = 5，r = e⁻²。
  O_acc = **e⁻²**·(e⁻²·v₁ + v₂) + 1·v₃ + e⁻³·v₄
        = e⁻⁴·v₁ + e⁻²·v₂ + 1·v₃ + e⁻³·v₄ ← 和全局 max=5 直接算完全一致。
  l 同理。之前「欠」的因子被 r 一次性补齐。

这就是「为什么敢立刻 softmax、立刻 PV」：PV 乘进去的 P 可能带着偏小的
max，但未来的 rescale 会把**整个 oaccu**修正回来。本 kernel 还把这个
rescale 乘法拆成对儿塞在 PV 的相邻 mfma 之间（common 头 L303），零额外
暴露时间。

### 3. 为什么 p_comp 这么小就够

因为 S 的生命周期只有「一个 tile 内」：本 tile 64 列分数 → 原地 exp →
pack bf16 → PV 吃掉 → 灭。跨 tile 存活的只有 O_acc（128 个 VGPR）+
m、l（各 1 个标量寄存器）。[16, T] 的完整 S 从不存在。
p_comp = 64 列 × 16 行 ÷ 64 lane = 16 VGPR，正好一个 tile。

### 4. row_max 具体怎么求（本 kernel 三步，L786–817）

回忆 lane 布局：lane l 持有 q 行 = l%16、本 tile 里 16 个 KV 列
（4 列一组、组间隔 16）。

1. **lane 内**：`max_16<>()` 对自己 16 个 p_comp 寄存器求 max
   → 该 lane 覆盖的 16 列的局部 max（纯 VALU，无跨线程）。
2. **lane 间**：`warp_reduce<MaxFunctor>`，只在
   {l, l+16, l+32, l+48} 这 4 个 lane 之间归约（stop_stride=15，
   swizzle 实现）——这 4 个 lane 正好持有**同一 q 行**的 4 份 16 列
   → 得到本 tile 完整 64 列的 `local_max`，且 4 个 lane 各留一份副本。
3. **和 running max 合并**：
   - 首 iter：`row_max = local_max`，无 rescale；
   - 否则：`local_max − row_max > 8.0`（kRescaleThreshold）才认为需要
     rescale，且用 `ballot` 升格成整个 wave 的一致决定（因为 oaccu 的
     rescale 是全 wave 一起做的寄存器扫乘）；不超阈值就**故意用旧的
     （偏小的）row_max 接着算**——exp(s − 旧max) 最多偏大 e⁸ ≈ 2981 倍，
     离 fp32 的 e⁸⁸ 溢出墙很远，而分子分母用的是同一个 m，数学上仍然
     精确，只是省掉了那次全 oaccu 扫乘（长上下文实测 ~3%）。

一句话总结：**dim 轴在 mfma 累加里早就收完了；T 轴不需要先看全局，
online softmax 用「先欠账、后补乘」把它变成了流式的**——所以 QK 一完就能
softmax，softmax 一完就能 PV，p_comp 只需要装一个 tile。

→ 深挖：spec Ch.9（softmax 各函数逐行）、Ch.10.4.1（kDoRescale 两种实例）。

## 2026-08-27 Q&A(2)：pinned 寄存器怎么用——去掉流水线的直筒版伪代码

> 问题原文：oaccu / q_vgpr / p_comp / p_mfma / pv_v_0 / pv_v_2 / k_2 这些都是
> 怎么用的？先不考虑流水线，排一下顺序。另外三个猜测求证：
> ① g2s q_nope + s2r q_rope 直接读到 register？
> ② q_nope 反量化后写进 q_vgpr，正好多少个 reg？
> ③ g2s kv (fp8 with scale) 之后怎么转换来着？

### 先回答三个猜测

① **q_nope 猜对了一半，q_rope 猜反了半步**：
   - q_nope：确实先 **g2s**（`buffer_load_lds_b128` 原始 fp8 → 每 warp 私有
     staging LDS），但这只是中转——之后还要 **s2r + 反量化** 才进 q_vgpr。
     走 LDS 一趟是为了把 lane 顺序摆成 mfma A 操作数要的样子。
   - q_rope：不是 s2r 直读，是 **g2r**（16 B load 直达寄存器）→ 乘温度 →
     **r2s** 写进 Q-LDS（写地址里做 sb8 置换 + 半行 swap，`p2_load_rope_chunk`）
     → prologue 里再 **s2r** 到 v120–127。绕 LDS 一圈唯一的目的是做置换。

② **q_nope 占 56 个，加 rope 8 个正好 64 = 填满 q_vgpr**：
   448 列 × 16 行 × 2 B(bf16) ÷ 64 lane ÷ 4 B = **56**（v64–v119，
   7 个 64 列 chunk × 8 reg）；rope 64 列 → 8 个（v120–v127）。

③ **KV 的转换：一条 gfx950 新指令搞定**——`v_cvt_scalef32_pk_bf16_fp8`：
   2×fp8 → 2×bf16 **顺带乘一个 float scale**，scale = `e8m0_to_f32(那 1 字节)`
   （2^e 展开）。搬运通路是混合的：strip 0/1 走 **g2r**（`buffer_load_dwordx4`
   "carrier"，16 fp8/lane 直达寄存器），strip 2/3 走 **g2s**（`buffer_load_lds`
   → staging，再 s2r 出来）；两路都在寄存器里做 scaled-cvt，然后 **r2s**
   `ds_write` 进 KV pong。两个注意点：
   - KV 的 cvt **不乘** softmax_scale——温度只折进 Q（乘一次就够）；
   - KV RoPE 是唯一纯 DMA 的：本来就是 bf16，`buffer_load_lds` 从 rope buffer
     **直达 pong**，不过寄存器、不转换。

### 直筒版伪代码（单 warp 视角，流水线全部拉直）

```python
# 记号: g=vmem  s=LDS  r=VGPR；"16 行"都指本 warp 自己那 16 行 Q / 16 行 KV band

# ========== 0. Prologue: 装 Q（每个 work item 一次） ==========
for c in range(7):                            # Q NoPE: 448 列 = 7 个 64 列 chunk
    g2s: buffer_load_lds_b128   q_nope[16, 64c:64c+64](fp8) -> staging
    g2r: buffer_load            该 chunk 的 e8m0 scale 字节  -> s_dw
    s2r: ds_read_b64            staging -> 8 fp8/lane
    cvt: v_cvt_scalef32_pk_bf16_fp8(fp8, e8m0(s_dw) * sm_scale*log2e)
         -> q_vgpr[v64+8c : v64+8c+8]         # 反量化+乘温度一条指令完成
# 7 chunk × 8 reg = v64..v119

g2r: q_rope(bf16) -> u32x4                    # Q RoPE: 64 列
r  : scale_bf16_pair(·, sm_scale*log2e)       # rope 也要乘温度
r2s: ds_write_b128 -> Q-LDS                   # 地址里做 sb8 置换
s2r: load_q_lds_to_gpr -> q_vgpr[v120:v128]
# 至此 v64..v127 = 完整 Q[16,512] bf16, 主循环只读; Q-LDS 报废

row_max, row_sum_e = -inf, 0                  # 每 lane 标量; oaccu 首个 PV 初始化

# ========== 1. 逐个 64 行 KV tile ==========
for tile in kv_tiles:
    # ---- 1a. 搬运: fp8 KV -> bf16 KV pong(LDS), 本 warp 只管 16行×256列 band ----
    row = resolve_row_kv_ld(tile)             # 查 page 表 -> 物理行号
    g2r: buffer_load_dwordx4  strip0,1 fp8 -> carrier寄存器   (+各1字节 scale)
    g2s: buffer_load_lds      strip2(,3 仅Lo) fp8 -> staging  (+scale)
    if Hi: g2s: buffer_load_lds  kv_rope(bf16) -> pong 直达    # 纯 DMA
    for strip in 自己的 fp8 strips:
        r  : fp8 = carrier 或 ds_read(staging)
        cvt: v_cvt_scalef32_pk_bf16_fp8(fp8, e8m0(scale))     # 不乘温度!
        r2s: ds_write -> pong[本 band 位置]
    barrier()                                 # 8 warp 拼齐 64×512 bf16 tile

    # ---- 1b. QK: p_comp[64,16] = K_tile @ Q^T, 512 dim 全部收缩 ----
    for c in range(16):                       # dim 512, 每段 32
        for rg in range(4):                   # 64 kv 行, 每组 16
            s2r: ds_read_b128  pong K 片段 -> k_ring        # v36/v40/v44 轮转
            mma: v_mfma_f32_16x16x32_bf16(
                     p_comp[rg] += k_ring @ q_vgpr[v64+4c : +4])
                     # p_comp = v48..v63: 4 个 16×16 fp32 累加器
    # 循环结束: p_comp = 本 tile 完整精确的 S_tile（dim 已收完, 见上一篇日志）

    # ---- 1c. softmax(原地) + pack ----
    mask -> max_16 -> warp_reduce -> 更新 row_max/rescale -> exp2 -> row_sum_e
    pack: v_cvt_pk_bf16_f32  p_comp(16 fp32) -> p_mfma[v48:v56](16 bf16)
    # p_comp 的 fp32 从此报废; v56..v63 空出来

    # ---- 1d. PV: oaccu[16,512] += P[16,64] @ V_tile[64,512] ----
    if do_rescale: oaccu *= rescale           # 实际拆成小段塞在 mfma 之间
    for t in range(32):                       # 512 输出列, 每段 16
        for half in (A, B):                   # kv 行 0:32 / 32:64
            s2r: ds_read_b64_tr_b16  pong 转置读 V^T 片段
                 -> pv_v 四槽轮转              # v60(pv_v_0)/v36/v40/v44 —— 借 k_ring!
            mma: v_mfma_f32_16x16x32_bf16(
                     oaccu[v128+4t : +4] += pv_v @ p_mfma[half])

# ========== 2. Epilogue(最后一个 tile 之后) ==========
row_sum_e += exp2(sink*log2e - row_max)       # 仅 final / 最后一个 split
r  : oaccu *= 1/row_sum_e                     # v128..v255 全扫
r2s: ds_write  oaccu -> O bounce LDS（盖在 next pong 上, 做 sb8 逆置换）
s2r+g: ds_read -> buffer_store -> final_output(bf16) 或 split_output(fp32)+LSE
```

### 寄存器 ↔ 步骤对照

| 寄存器 | 写它的步骤 | 读它的步骤 | 生命周期 |
|---|---|---|---|
| `q_vgpr` v64–127 | 0（装 Q） | 1b 每条 mma 的 B 操作数 | 整个 work item，只写一次 |
| `k_0/1/2` v36–47 | 1b ds_read | 1b 紧跟的 mma | 一对 mfma 内（3 槽轮转） |
| `p_comp` v48–63 | 1b mma 累加 | 1c softmax 原地 | 一个 tile |
| `p_mfma` v48–55 | 1c pack | 1d PV 的 B 操作数 | 1c→1d（overlay 在 p_comp 上） |
| `pv_v_0..3` v60,36,40,44 | 1d 转置 ds_read | 1d 紧跟的 mma | 一对 mfma 内（4 槽轮转，借 QK 的 k_ring + p_comp 顶部） |
| `oaccu` v128–255 | 1d mma 累加 | 2 归一化+写出 | 整个 work item（跨 tile 存活） |

一眼看懂复用逻辑：**QK 和 PV 永远不同时跑**，所以 v36–47 这 12 个寄存器在
QK 期间叫 k_ring（装 K）、PV 期间叫 pv_v（装 V）；p_comp 在 softmax pack 完
之后 fp32 内容报废，低 8 个被 bf16 的 P 覆盖（p_mfma）、顶上 4 个借给 V 槽。
真正需要跨 tile 活着的只有 q_vgpr、oaccu 和 (row_max, row_sum_e) 两个标量——
这正好呼应上一篇日志的结论。

→ 深挖：spec Ch.6–7（Q 两阶段装载与 sb8 置换）、Ch.8.5–8.6（KV cvt+store 逐行）。