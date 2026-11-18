//! O(n^3) 匈牙利算法（最小化，稠密方阵，势能法）。
//!
//! 等价性说明：Python 侧用 scipy.optimize.linear_sum_assignment，
//! 其内部为 Jonker-Volgenant 算法并对方阵补零；矩形代价矩阵
//! "每行分配一个不同列"与"补零成方阵"在数学上等价（补零行/列吃掉
//! 剩余配额，不改变真实行列的最优配对方案）。本实现先把矩形矩阵
//! 补零成 n=max(n_rows,n_cols) 方阵，再用经典 O(n^3) 势能法求解，
//! tie-break 策略为：行按顺序处理、候选列严格小于才更新 —— 完全
//! 确定。门控项 1e6 远大于正常代价与 0 填充，保证被门控的配对
//! 不会进入最终结果（Python 侧再用 cost<0.999 过滤）。

/// 求解最小代价分配。cost 为 n_rows x n_cols 矩阵。
/// 返回 (row, col) 配对列表（含补零产生的虚拟行列已被剔除）。
pub fn hungarian(cost: &[Vec<f64>]) -> Vec<(usize, usize)> {
    let n_rows = cost.len();
    let n_cols = if n_rows > 0 { cost[0].len() } else { 0 };
    let n = n_rows.max(n_cols);
    if n == 0 {
        return Vec::new();
    }
    // 补零成方阵
    let mut a = vec![vec![0.0f64; n]; n];
    for (i, row) in cost.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            a[i][j] = v;
        }
    }
    let assign = hungarian_square(&a);
    let mut out = Vec::new();
    for (r, &c) in assign.iter().enumerate() {
        if r < n_rows && c < n_cols {
            out.push((r, c));
        }
    }
    out
}

/// 经典势能法匈牙利（1-based 内部索引，方阵）。
/// 返回每行被分配的列。
fn hungarian_square(a: &[Vec<f64>]) -> Vec<usize> {
    let n = a.len();
    let inf = f64::INFINITY;
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1]; // p[j] = 分配给列 j 的行
    let mut way = vec![0usize; n + 1];
    let mut assignment = vec![usize::MAX; n];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (j, used_j) in used.iter().enumerate() {
                if *used_j {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        // 增广路径回退
        while j0 > 0 {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        }
    }
    for j in 1..=n {
        if p[j] > 0 {
            assignment[p[j] - 1] = j - 1;
        }
    }
    assignment
}
