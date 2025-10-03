use std::path::PathBuf;

/// Returns the path to the sample CSV relative to the crate root.
pub fn sample_csv_path() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .expect("Cannot determine executable path");

    let crate_root = exe_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("Cannot find crate root");

    crate_root.join("data").join("data.csv")
}
