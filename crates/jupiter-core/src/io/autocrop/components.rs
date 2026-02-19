use ndarray::Array2;

/// Statistics for a single connected component.
#[derive(Clone, Debug)]
pub struct ComponentStats {
    /// Unique label for this component.
    pub label: u32,
    /// Number of pixels in the component.
    pub area: usize,
    /// Bounding box: (min_row, max_row, min_col, max_col).
    pub bbox: (usize, usize, usize, usize),
}

/// Perform connected component analysis on a binary mask using two-pass
/// labeling with union-find. Uses 4-connectivity (left and upper neighbors).
///
/// Returns component statistics sorted by area descending (largest first).
pub fn connected_components(mask: &Array2<bool>) -> Vec<ComponentStats> {
    let (h, w) = mask.dim();
    if h == 0 || w == 0 {
        return Vec::new();
    }

    let mut labels = Array2::<u32>::zeros((h, w));
    let mut next_label: u32 = 1;
    // Union-find parent array. Index 0 unused; labels start at 1.
    let mut parent: Vec<u32> = vec![0; h * w / 2 + 2];

    // Pass 1: assign provisional labels.
    for row in 0..h {
        for col in 0..w {
            if !mask[[row, col]] {
                continue;
            }

            let up = if row > 0 { labels[[row - 1, col]] } else { 0 };
            let left = if col > 0 { labels[[row, col - 1]] } else { 0 };

            match (up > 0, left > 0) {
                (false, false) => {
                    // New label.
                    if next_label as usize >= parent.len() {
                        parent.resize(parent.len() * 2, 0);
                    }
                    parent[next_label as usize] = next_label;
                    labels[[row, col]] = next_label;
                    next_label += 1;
                }
                (true, false) => {
                    labels[[row, col]] = up;
                }
                (false, true) => {
                    labels[[row, col]] = left;
                }
                (true, true) => {
                    let smaller = up.min(left);
                    let larger = up.max(left);
                    labels[[row, col]] = smaller;
                    if smaller != larger {
                        union(&mut parent, smaller, larger);
                    }
                }
            }
        }
    }

    // Flatten parent references.
    for i in 1..next_label as usize {
        parent[i] = find(&parent, i as u32);
    }

    // Pass 2: resolve labels and collect stats.
    let mut stats_map = std::collections::HashMap::<u32, ComponentStats>::new();

    for row in 0..h {
        for col in 0..w {
            let lbl = labels[[row, col]];
            if lbl == 0 {
                continue;
            }
            let root = parent[lbl as usize];

            let entry = stats_map.entry(root).or_insert(ComponentStats {
                label: root,
                area: 0,
                bbox: (row, row, col, col),
            });

            entry.area += 1;
            entry.bbox.0 = entry.bbox.0.min(row);
            entry.bbox.1 = entry.bbox.1.max(row);
            entry.bbox.2 = entry.bbox.2.min(col);
            entry.bbox.3 = entry.bbox.3.max(col);
        }
    }

    let mut components: Vec<ComponentStats> = stats_map.into_values().collect();
    components.sort_unstable_by(|a, b| b.area.cmp(&a.area));
    components
}

/// Returns true if the component's bounding box touches any edge of the image.
pub fn touches_border(bbox: (usize, usize, usize, usize), height: usize, width: usize) -> bool {
    let (min_row, max_row, min_col, max_col) = bbox;
    min_row == 0 || max_row >= height - 1 || min_col == 0 || max_col >= width - 1
}

fn find(parent: &[u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        x = parent[x as usize];
    }
    x
}

fn union(parent: &mut [u32], a: u32, b: u32) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        // Merge larger root into smaller root to keep labels consistent.
        let (small, big) = if ra < rb { (ra, rb) } else { (rb, ra) };
        parent[big as usize] = small;
    }
}
