#[derive(Debug, Clone, Copy)]
pub enum ColumnType {
    Int64,
    Float64,
    Str,
}

#[derive(Debug, Clone)]
pub enum Column {
    Int64(Vec<Vec<i64>>),
    Float64(Vec<Vec<f64>>),
    Str(Vec<Vec<(usize, usize)>>), // Absolute offsets into mmap
}

impl Column {
    pub fn new_int64() -> Self {
        Column::Int64(Vec::new())
    }

    pub fn new_float64() -> Self {
        Column::Float64(Vec::new())
    }

    pub fn new_str() -> Self {
        Column::Str(Vec::new())
    }

    pub fn push_chunk_int64(&mut self, chunk: Vec<i64>) {
        match self {
            Column::Int64(chunks) => chunks.push(chunk),
            _ => panic!("Type mismatch"),
        }
    }

    pub fn push_chunk_float64(&mut self, chunk: Vec<f64>) {
        match self {
            Column::Float64(chunks) => chunks.push(chunk),
            _ => panic!("Type mismatch"),
        }
    }

    pub fn push_chunk_str(&mut self, chunk: Vec<(usize, usize)>) {
        if let Column::Str(chunks) = self {
            chunks.push(chunk)
        } else {
            panic!("Wrong type")
        }

        // match self {
        //     Column::Str(chunks) => chunks.push(chunk),
        //     _ => panic!("Type mismatch"),
        // }
    }

    // Efficient iteration
    pub fn iter_i64(&self) -> impl Iterator<Item = i64> + '_ {
        if let Column::Int64(chunks) = self {
            chunks.iter().flat_map(|chunk| chunk.iter().copied())
        } else {
            panic!("Wrong type")
        }
        // match self {
        //     Column::Int64(chunks) => {
        //         Box::new(chunks.iter().flat_map(|chunk| chunk.iter().copied()))
        //             as Box<dyn Iterator<Item = i64> + '_>
        //     }
        //     _ => panic!("Wrong type"),
        // }
    }

    pub fn iter_f64(&self) -> impl Iterator<Item = f64> + '_ {
        if let Column::Float64(chunks) = self {
            chunks.iter().flat_map(|chunk| chunk.iter().copied())
        } else {
            panic!("Wrong type")
        }
        // match self {
        //     Column::Float64(chunks) => {
        //         Box::new(chunks.iter().flat_map(|chunk| chunk.iter().copied()))
        //             as Box<dyn Iterator<Item = f64> + '_>
        //     }
        //     _ => panic!("Wrong type"),
        // }
    }

    pub fn iter_str(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        match self {
            Column::Str(chunks) => Box::new(chunks.iter().flat_map(|chunk| chunk.iter().copied()))
                as Box<dyn Iterator<Item = (usize, usize)> + '_>,
            _ => panic!("Wrong type"),
        }
    }

    // Random access
    pub fn get_i64(&self, idx: usize) -> Option<i64> {
        match self {
            Column::Int64(chunks) => {
                let mut remaining = idx;
                for chunk in chunks {
                    if remaining < chunk.len() {
                        return Some(chunk[remaining]);
                    }
                    remaining -= chunk.len();
                }
                None
            }
            _ => panic!("Wrong type"),
        }
    }

    pub fn total_len(&self) -> usize {
        match self {
            Column::Int64(chunks) => chunks.iter().map(|c| c.len()).sum(),
            Column::Float64(chunks) => chunks.iter().map(|c| c.len()).sum(),
            Column::Str(chunks) => chunks.iter().map(|c| c.len()).sum(),
        }
    }

    pub fn flatten_in_place(&mut self) {
        match self {
            Column::Int64(chunks) => {
                if chunks.len() <= 1 {
                    return; // Already flat
                }

                // Take ownership of chunks, leaving empty vec
                let mut owned_chunks = std::mem::take(chunks);

                // Use the first chunk as the base (it's already allocated)
                let mut flattened = owned_chunks.swap_remove(0);

                // Calculate total capacity needed
                let total: usize = owned_chunks.iter().map(|c| c.len()).sum();
                flattened.reserve(total);

                // Extend from remaining chunks (this moves data, no copy)
                for chunk in owned_chunks {
                    flattened.extend(chunk);
                }

                // Put the flattened vec back as a single chunk
                chunks.push(flattened);
            }
            Column::Float64(chunks) => {
                if chunks.len() <= 1 {
                    return;
                }

                let mut owned_chunks = std::mem::take(chunks);
                let mut flattened = owned_chunks.swap_remove(0);
                let total: usize = owned_chunks.iter().map(|c| c.len()).sum();
                flattened.reserve(total);

                for chunk in owned_chunks {
                    flattened.extend(chunk);
                }

                chunks.push(flattened);
            }
            Column::Str(chunks) => {
                if chunks.len() <= 1 {
                    return;
                }

                let mut owned_chunks = std::mem::take(chunks);
                let mut flattened = owned_chunks.swap_remove(0);
                let total: usize = owned_chunks.iter().map(|c| c.len()).sum();
                flattened.reserve(total);

                for chunk in owned_chunks {
                    flattened.extend(chunk);
                }

                chunks.push(flattened);
            }
        }
    }
}
