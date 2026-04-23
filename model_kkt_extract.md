# 第三、四章数学模型与 KKT 推导提取

本文模型面向协同边缘计算中的冗余分阶段服务放置、阶段级任务卸载、节点计算资源分配和边缘链路带宽分配。第三章建立原始混合整数非线性规划问题 `P1`，第四章在固定离散决策 `x,y` 后，将连续资源分配问题分解为节点级与链路级凸子问题，并通过 KKT 得到闭式最优解。

## 1. 集合、参数与变量

### 集合与索引

- 边缘节点集合：`M = {1, ..., M}`，节点记为 `E_m`。
- 服务类型集合：`I = {1, ..., I}`。
- 服务 `i` 被拆分为 `J_i` 个有序阶段：`S_{i,1}, ..., S_{i,J_i}`。
- 节点 `E_m` 覆盖 `N_m` 个用户，用户记为 `U_{m,n}`。
- 用户 `U_{m,n}` 请求服务 `i` 的第 `j` 阶段任务记为 `K_{m,n,i,j}`。
- 阶段链：

```math
K_{m,n,i} = \{K_{m,n,i,1}, K_{m,n,i,2}, ..., K_{m,n,i,J_i}\}.
```

```math
\mathcal{E}_{m,n,i}
= \{(K_{m,n,i,j}, K_{m,n,i,j+1}) \mid j=1,...,J_i-1\}.
```

### 系统参数

- `a_{m,m'}`：节点 `E_m` 与 `E_{m'}` 是否直连。
- `R_{m,m'}`：链路 `(E_m,E_{m'})` 的最大传输速率。
- `F_m`：节点 `E_m` 的计算资源上限。
- `A_m, B_m`：节点 `E_m` 的存储和内存容量。
- `a_{i,j}, b_{i,j}`：阶段服务 `S_{i,j}` 的存储、内存占用。
- `C_{m,n,i,j}`：任务阶段 `K_{m,n,i,j}` 的计算 workload。
- `D_{m,n,i}`：从归属节点到第一阶段执行节点的输入数据量。
- `D_{m,n,i,j}`：服务 `i` 第 `j` 阶段输出的中间数据量。

### 决策变量

- 服务放置变量：

```math
x_{i,j,m'} \in \{0,1\}.
```

`x_{i,j,m'}=1` 表示节点 `E_{m'}` 放置阶段服务 `S_{i,j}`。

- 阶段级卸载变量：

```math
y_{m,n,i,j,m'} \in \{0,1\}.
```

`y_{m,n,i,j,m'}=1` 表示任务阶段 `K_{m,n,i,j}` 在节点 `E_{m'}` 执行。

- 节点计算资源变量：

```math
f_{m',m,n,i,j} \ge 0.
```

表示节点 `E_{m'}` 分配给任务阶段 `K_{m,n,i,j}` 的计算资源。

- 链路速率变量：

```math
r_{m,n,i,0,m,m'} \ge 0,
```

表示归属节点 `E_m` 到第一阶段执行节点 `E_{m'}` 的传输速率。

```math
r_{m,n,i,j,m',m''} \ge 0,
```

表示阶段 `j` 在 `E_{m'}` 执行、阶段 `j+1` 在 `E_{m''}` 执行时，中间结果从 `E_{m'}` 到 `E_{m''}` 的传输速率。

## 2. 第三章：原始数学模型

### 2.1 服务放置约束

二进制放置：

```math
x_{i,j,m'} \in \{0,1\},
\quad \forall i,j,m'.
```

内存容量：

```math
\sum_{i=1}^{I}\sum_{j=1}^{J_i} x_{i,j,m'} b_{i,j}
\le B_{m'},
\quad \forall m'.
```

存储容量：

```math
\sum_{i=1}^{I}\sum_{j=1}^{J_i} x_{i,j,m'} a_{i,j}
\le A_{m'},
\quad \forall m'.
```

### 2.2 阶段级卸载约束

放置与卸载一致性：

```math
y_{m,n,i,j,m'} \le x_{i,j,m'},
\quad \forall m,n,i,j,m'.
```

每个阶段只能被一个节点执行：

```math
\sum_{m'=1}^{M} y_{m,n,i,j,m'} = 1,
\quad \forall m,n,i,j.
```

完整处理条件：

```math
\sum_{j=1}^{J_i}\sum_{m'=1}^{M} y_{m,n,i,j,m'} = J_i,
\quad \forall m,n,i.
```

该约束在逐阶段 one-hot 约束成立时是冗余校验，但实现中可以作为一致性检查。

执行节点索引可由 one-hot 卸载变量得到：

```math
\pi_{m,n,i,j}
= \sum_{m'=1}^{M} m' y_{m,n,i,j,m'}.
```

完整执行路径：

```math
\Pi_{m,n,i}
= (\pi_{m,n,i,1}, \pi_{m,n,i,2}, ..., \pi_{m,n,i,J_i}).
```

### 2.3 计算资源约束与计算时延

有效计算资源分配：

```math
0 \le f_{m',m,n,i,j}
\le x_{i,j,m'}y_{m,n,i,j,m'}F_{m'},
\quad \forall m,m',n,i,j.
```

节点计算资源容量：

```math
\sum_{m=1}^{M}\sum_{n=1}^{N_m}\sum_{i=1}^{I}\sum_{j=1}^{J_i}
f_{m',m,n,i,j}
\le F_{m'},
\quad \forall m'.
```

阶段计算时延：

```math
t^{comp}_{m',m,n,i,j}
=
\frac{x_{i,j,m'}y_{m,n,i,j,m'}C_{m,n,i,j}}
{f_{m',m,n,i,j}}.
```

请求 `(m,n,i)` 的总计算时延：

```math
T^{comp}_{m,n,i}
=
\sum_{j=1}^{J_i}\sum_{m'=1}^{M}
\frac{x_{i,j,m'}y_{m,n,i,j,m'}C_{m,n,i,j}}
{f_{m',m,n,i,j}}.
```

### 2.4 链路速率约束与传输时延

物理链路 `(E_m,E_{m'})` 的容量约束：

```math
\sum_{n=1}^{N_m}\sum_{i=1}^{I}
r_{m,n,i,0,m,m'}
+
\sum_{\bar m=1}^{M}\sum_{\bar n=1}^{N_{\bar m}}\sum_{i=1}^{I}
\sum_{j=1}^{J_i-1}
r_{\bar m,\bar n,i,j,m,m'}
\le R_{m,m'},
\quad \forall m \ne m'.
```

非负速率：

```math
r_{m,n,i,0,m,m'} \ge 0,
\quad
r_{m,n,i,j,m',m''} \ge 0.
```

若两个相邻阶段在同一节点执行，则无需边缘间传输，对应速率变量自然为 `0`。

第一阶段跨节点传输时延：

```math
T^{tran,0}_{m,n,i}
=
\sum_{\substack{m'=1\\m'\ne m}}^{M}
\frac{y_{m,n,i,1,m'}D_{m,n,i}}
{r_{m,n,i,0,m,m'}}.
```

相邻阶段跨节点传输时延：

```math
T^{tran,j}_{m,n,i}
=
\sum_{m'=1}^{M}\sum_{\substack{m''=1\\m''\ne m'}}^{M}
\frac{y_{m,n,i,j,m'}y_{m,n,i,j+1,m''}D_{m,n,i,j}}
{r_{m,n,i,j,m',m''}},
\quad j=1,...,J_i-1.
```

总传输时延：

```math
T^{tran}_{m,n,i}
= T^{tran,0}_{m,n,i}
+ \sum_{j=1}^{J_i-1}T^{tran,j}_{m,n,i}.
```

### 2.5 请求时延与系统总目标

完整端到端时延：

```math
\tilde T_{m,n,i}
= t^{acc}_{m,n,i}
+ T^{tran}_{m,n,i}
+ T^{comp}_{m,n,i}.
```

其中 `t^{acc}_{m,n,i}` 是用户到归属边缘节点的接入时延，不参与优化。优化时只需最小化：

```math
T_{m,n,i}
= T^{tran}_{m,n,i}+T^{comp}_{m,n,i}.
```

系统总时延：

```math
T_{total}
=
\sum_{m=1}^{M}\sum_{n=1}^{N_m}\sum_{i=1}^{I}
\left(T^{tran}_{m,n,i}+T^{comp}_{m,n,i}\right).
```

原始问题：

```math
\text{P1:}\quad
\min_{x,y,f,r} T_{total}
\quad
\text{s.t. placement, offloading, computing, rate constraints.}
```

`P1` 同时含二进制变量 `x,y` 和连续变量 `f,r`，目标中含 `C/f` 与 `D/r` 形式，因此是混合整数非线性规划问题。

## 3. 第四章：固定离散决策后的连续资源分配

固定上层放置和卸载决策 `x,y` 后，只优化连续变量 `f,r`：

```math
\text{P2:}\quad
\min_{f,r}
\sum_{m,n,i,j,m'}
\frac{x_{i,j,m'}y_{m,n,i,j,m'}C_{m,n,i,j}}
{f_{m',m,n,i,j}}
+
\sum_{m,n,i,m'\ne m}
\frac{y_{m,n,i,1,m'}D_{m,n,i}}
{r_{m,n,i,0,m,m'}}
+
\sum_{m,n,i,j,m'\ne m''}
\frac{y_{m,n,i,j,m'}y_{m,n,i,j+1,m''}D_{m,n,i,j}}
{r_{m,n,i,j,m',m''}}.
```

约束为第三章中的连续资源约束：

```math
0 \le f_{m',m,n,i,j}
\le x_{i,j,m'}y_{m,n,i,j,m'}F_{m'},
```

```math
\sum_{m,n,i,j}f_{m',m,n,i,j}\le F_{m'},
```

```math
r_{m,n,i,0,m,m'}\ge 0,\quad
r_{m,n,i,j,m',m''}\ge 0,
```

```math
\sum r \le R_{m,m'}.
```

在 `x,y` 固定后，`c/f` 与 `d/r` 在正域上严格凸，约束为线性，因此 `P2` 是凸问题，并可分解为：

- 节点级计算资源分配子问题；
- 链路级带宽分配子问题。

## 4. 节点计算资源分配的 KKT 推导

固定任意节点 `E_{m'}`，只考虑满足 `x_{i,j,m'}y_{m,n,i,j,m'}=1` 的活跃任务阶段。

节点级子问题：

```math
\text{P2-1:}\quad
\min_{f}
\sum_{m=1}^{M}\sum_{n=1}^{N_m}\sum_{i=1}^{I}\sum_{j=1}^{J_i}
\frac{x_{i,j,m'}y_{m,n,i,j,m'}C_{m,n,i,j}}
{f_{m',m,n,i,j}}
```

```math
\text{s.t.}\quad
\sum_{m,n,i,j}f_{m',m,n,i,j}\le F_{m'},
\quad
f_{m',m,n,i,j}\ge 0.
```

若节点 `E_{m'}` 至少承载一个活跃任务阶段，则最优点处计算资源容量约束取等号：

```math
\sum_{m,n,i,j}f_{m',m,n,i,j}=F_{m'}.
```

理由：目标项 `C/f` 随 `f` 单调递减，若还有剩余计算资源，则可以继续增加某个活跃任务的 `f` 并降低目标值。

### 4.1 拉格朗日函数

令 `lambda_{m'} >= 0` 为节点容量约束的拉格朗日乘子：

```math
L_{m'}
=
\sum_{m,n,i,j}
\frac{x_{i,j,m'}y_{m,n,i,j,m'}C_{m,n,i,j}}
{f_{m',m,n,i,j}}
+
\lambda_{m'}
\left(
\sum_{m,n,i,j}f_{m',m,n,i,j}-F_{m'}
\right).
```

### 4.2 KKT 条件

对任意活跃任务阶段：

```math
\frac{\partial L_{m'}}{\partial f_{m',m,n,i,j}}
=
-
\frac{x_{i,j,m'}y_{m,n,i,j,m'}C_{m,n,i,j}}
{f_{m',m,n,i,j}^2}
+ \lambda_{m'}
=0.
```

活跃时 `x_{i,j,m'}y_{m,n,i,j,m'}=1`，因此：

```math
f_{m',m,n,i,j}
=
\sqrt{\frac{C_{m,n,i,j}}{\lambda_{m'}}}.
```

定义节点 `E_{m'}` 的平方根聚合负载：

```math
\Gamma_{m'}
=
\sum_{m,n,i,j}
x_{i,j,m'}y_{m,n,i,j,m'}
\sqrt{C_{m,n,i,j}}.
```

由紧容量约束可得：

```math
\lambda_{m'}
=
\left(\frac{\Gamma_{m'}}{F_{m'}}\right)^2.
```

### 4.3 闭式最优解

活跃任务阶段的最优计算资源：

```math
f^*_{m',m,n,i,j}
=
\frac{F_{m'}\sqrt{C_{m,n,i,j}}}{\Gamma_{m'}}.
```

非活跃任务阶段：

```math
f^*_{m',m,n,i,j}=0.
```

节点 `E_{m'}` 的最小计算时延：

```math
T^{comp,*}_{m'}
=
\frac{\Gamma_{m'}^2}{F_{m'}}.
```

结论：固定 `x,y` 时，节点计算资源按任务计算量 `C` 的平方根比例分配。

## 5. 链路带宽分配的 KKT 推导

固定任意有向链路 `(E_{m'},E_{m''})`，其中 `m' != m''`。该链路承载两类流：

- 第一阶段传输：用户归属节点为 `E_{m'}`，第一阶段执行节点为 `E_{m''}`；
- 阶段间传输：阶段 `j` 在 `E_{m'}` 执行，阶段 `j+1` 在 `E_{m''}` 执行。

链路级子问题：

```math
\text{P2-2:}\quad
\min_r
\sum_{n=1}^{N_{m'}}\sum_{i=1}^{I}
\frac{y_{m',n,i,1,m''}D_{m',n,i}}
{r_{m',n,i,0,m',m''}}
+
\sum_{m=1}^{M}\sum_{n=1}^{N_m}\sum_{i=1}^{I}\sum_{j=1}^{J_i-1}
\frac{y_{m,n,i,j,m'}y_{m,n,i,j+1,m''}D_{m,n,i,j}}
{r_{m,n,i,j,m',m''}}.
```

```math
\text{s.t.}\quad
\sum_{n,i} r_{m',n,i,0,m',m''}
+
\sum_{m,n,i,j} r_{m,n,i,j,m',m''}
\le R_{m',m''},
```

```math
r_{m',n,i,0,m',m''}\ge 0,
\quad
r_{m,n,i,j,m',m''}\ge 0.
```

若链路 `(E_{m'},E_{m''})` 至少承载一个活跃传输流，则最优点处链路容量约束取等号：

```math
\sum r = R_{m',m''}.
```

理由同节点问题：每个传输时延项 `D/r` 随 `r` 单调递减。

### 5.1 拉格朗日函数

令 `eta_{m',m''} >= 0` 为链路容量约束的拉格朗日乘子：

```math
L_{m',m''}
=
\sum_{n,i}
\frac{y_{m',n,i,1,m''}D_{m',n,i}}
{r_{m',n,i,0,m',m''}}
+
\sum_{m,n,i,j}
\frac{y_{m,n,i,j,m'}y_{m,n,i,j+1,m''}D_{m,n,i,j}}
{r_{m,n,i,j,m',m''}}
+
\eta_{m',m''}
\left(
\sum r - R_{m',m''}
\right).
```

### 5.2 KKT 条件

第一阶段活跃传输流：

```math
-
\frac{y_{m',n,i,1,m''}D_{m',n,i}}
{r_{m',n,i,0,m',m''}^2}
+\eta_{m',m''}
=0.
```

活跃时 `y_{m',n,i,1,m''}=1`，因此：

```math
r_{m',n,i,0,m',m''}
=
\sqrt{\frac{D_{m',n,i}}{\eta_{m',m''}}}.
```

阶段间活跃传输流：

```math
-
\frac{y_{m,n,i,j,m'}y_{m,n,i,j+1,m''}D_{m,n,i,j}}
{r_{m,n,i,j,m',m''}^2}
+\eta_{m',m''}
=0.
```

活跃时：

```math
r_{m,n,i,j,m',m''}
=
\sqrt{\frac{D_{m,n,i,j}}{\eta_{m',m''}}}.
```

定义链路 `(E_{m'},E_{m''})` 的平方根聚合负载：

```math
S_{m',m''}
=
\sum_{n=1}^{N_{m'}}\sum_{i=1}^{I}
y_{m',n,i,1,m''}\sqrt{D_{m',n,i}}
+
\sum_{m=1}^{M}\sum_{n=1}^{N_m}\sum_{i=1}^{I}\sum_{j=1}^{J_i-1}
y_{m,n,i,j,m'}y_{m,n,i,j+1,m''}
\sqrt{D_{m,n,i,j}}.
```

由紧链路容量约束可得：

```math
\eta_{m',m''}
=
\left(\frac{S_{m',m''}}{R_{m',m''}}\right)^2.
```

### 5.3 闭式最优解

活跃第一阶段传输流：

```math
r^*_{m',n,i,0,m',m''}
=
\frac{R_{m',m''}\sqrt{D_{m',n,i}}}{S_{m',m''}}.
```

活跃阶段间传输流：

```math
r^*_{m,n,i,j,m',m''}
=
\frac{R_{m',m''}\sqrt{D_{m,n,i,j}}}{S_{m',m''}}.
```

非活跃传输流：

```math
r^*=0.
```

链路 `(E_{m'},E_{m''})` 的最小传输时延：

```math
T^{tran,*}_{m',m''}
=
\frac{S_{m',m''}^2}{R_{m',m''}}.
```

结论：固定 `x,y` 时，链路带宽按传输数据量 `D` 的平方根比例分配。

## 6. 项目实现要点

### 6.1 可直接实现的下层 KKT 模块

输入：

- `x[i,j,node]`
- `y[src,n,i,j,node]`
- `C[src,n,i,j]`
- `D0[src,n,i]`
- `Dstage[src,n,i,j]`
- `F[node]`
- `R[u,v]`

输出：

- `f_star[node,src,n,i,j]`
- `r0_star[src,n,i,src,dst]`
- `r_stage_star[src,n,i,j,u,v]`
- `T_comp_star`
- `T_tran_star`
- `T_total_star`

节点计算分配：

```text
for node u:
    active = {(src,n,i,j) | x[i,j,u] == 1 and y[src,n,i,j,u] == 1}
    Gamma = sum(sqrt(C[src,n,i,j]) for active)
    if Gamma == 0:
        f_star[u,...] = 0
    else:
        f_star[u,src,n,i,j] = F[u] * sqrt(C[src,n,i,j]) / Gamma
        node_delay = Gamma^2 / F[u]
```

链路带宽分配：

```text
for directed link (u,v), u != v:
    active_first = {(n,i) | src == u and y[u,n,i,1,v] == 1}
    active_stage = {(src,n,i,j) | y[src,n,i,j,u] == 1 and y[src,n,i,j+1,v] == 1}
    S = sum(sqrt(D0[u,n,i]) for active_first)
        + sum(sqrt(Dstage[src,n,i,j]) for active_stage)
    if S == 0:
        r_star[...] = 0
    else:
        r0_star[u,n,i,u,v] = R[u,v] * sqrt(D0[u,n,i]) / S
        r_stage_star[src,n,i,j,u,v] = R[u,v] * sqrt(Dstage[src,n,i,j]) / S
        link_delay = S^2 / R[u,v]
```

### 6.2 上层离散决策的可行性检查

实现时建议在调用 KKT 模块前检查：

- `x` 是否为二进制。
- 每个节点的内存、存储放置容量是否满足。
- `y` 是否为二进制。
- 每个任务阶段是否 one-hot。
- 是否满足 `y <= x`。
- 若 `a[u,v] == 0` 或 `R[u,v] == 0`，不允许生成从 `u` 到 `v` 的第一阶段或阶段间传输。
- 论文中的完整处理约束可作为 one-hot 后的冗余校验。

### 6.3 论文中需注意的疑似编号问题

- 第四章节点资源分配部分写“constraint (26) is tight”，按上下文应指节点容量约束 `(39)`。
- 第四章链路带宽分配部分写“constraint (36) is tight”，按上下文应指链路容量约束 `(49)`。
