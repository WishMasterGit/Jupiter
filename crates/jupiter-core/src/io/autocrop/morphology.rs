use ndarray::Array2;

/// Morphological opening (erosion followed by dilation) with a 3x3 square kernel.
///
/// Removes small isolated foreground pixels while preserving larger regions.
pub fn morphological_opening(mask: &Array2<bool>) -> Array2<bool> {
    let eroded = erode(mask);
    dilate(&eroded)
}

/// Binary erosion: a pixel stays true only if ALL pixels in its 3x3 neighborhood are true.
fn erode(mask: &Array2<bool>) -> Array2<bool> {
    let (h, w) = mask.dim();
    let mut result = Array2::from_elem((h, w), false);

    for row in 0..h {
        for col in 0..w {
            if !mask[[row, col]] {
                continue;
            }
            // Check all 9 neighbors (out-of-bounds treated as false).
            let mut all_true = true;
            for dr in -1..=1_i32 {
                for dc in -1..=1_i32 {
                    let nr = row as i32 + dr;
                    let nc = col as i32 + dc;
                    if nr < 0 || nr >= h as i32 || nc < 0 || nc >= w as i32 {
                        all_true = false;
                        break;
                    }
                    if !mask[[nr as usize, nc as usize]] {
                        all_true = false;
                        break;
                    }
                }
                if !all_true {
                    break;
                }
            }
            result[[row, col]] = all_true;
        }
    }

    result
}

/// Binary dilation: a pixel becomes true if ANY pixel in its 3x3 neighborhood is true.
fn dilate(mask: &Array2<bool>) -> Array2<bool> {
    let (h, w) = mask.dim();
    let mut result = Array2::from_elem((h, w), false);

    for row in 0..h {
        for col in 0..w {
            let mut any_true = false;
            for dr in -1..=1_i32 {
                for dc in -1..=1_i32 {
                    let nr = row as i32 + dr;
                    let nc = col as i32 + dc;
                    if nr >= 0
                        && nr < h as i32
                        && nc >= 0
                        && nc < w as i32
                        && mask[[nr as usize, nc as usize]]
                    {
                        any_true = true;
                        break;
                    }
                }
                if any_true {
                    break;
                }
            }
            result[[row, col]] = any_true;
        }
    }

    result
}
