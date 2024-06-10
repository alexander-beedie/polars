use std::path::{Path, PathBuf};

use polars_core::prelude::*;
use polars_io::cloud::CloudOptions;
use polars_io::ipc::IpcScanOptions;
use polars_io::utils::resolve_homedir;
use polars_io::RowIndex;

use crate::prelude::*;

#[derive(Clone)]
pub struct ScanArgsIpc {
    pub n_rows: Option<usize>,
    pub cache: bool,
    pub rechunk: bool,
    pub row_index: Option<RowIndex>,
    pub memory_map: bool,
    pub cloud_options: Option<CloudOptions>,
}

impl Default for ScanArgsIpc {
    fn default() -> Self {
        Self {
            n_rows: None,
            cache: true,
            rechunk: false,
            row_index: None,
            memory_map: true,
            cloud_options: Default::default(),
        }
    }
}

#[derive(Clone)]
struct LazyIpcReader {
    args: ScanArgsIpc,
    path: PathBuf,
    paths: Arc<[PathBuf]>,
}

impl LazyIpcReader {
    fn new(path: PathBuf, args: ScanArgsIpc) -> Self {
        Self {
            args,
            path: resolve_homedir(&path),
            paths: Arc::new([]),
        }
    }
}

impl LazyFileListReader for LazyIpcReader {
    fn finish(mut self) -> PolarsResult<LazyFrame> {
        if let Some(paths) = self.iter_paths()? {
            let paths = paths
                .into_iter()
                .collect::<PolarsResult<Arc<[PathBuf]>>>()?;
            self.paths = paths;
        }
        self.finish_no_glob()
    }

    fn finish_no_glob(self) -> PolarsResult<LazyFrame> {
        let args = self.args;

        let paths = if self.paths.is_empty() {
            Arc::new([self.path]) as Arc<[PathBuf]>
        } else {
            self.paths
        };

        let options = IpcScanOptions {
            memory_map: args.memory_map,
        };

        let mut lf: LazyFrame = DslBuilder::scan_ipc(
            paths,
            options,
            args.n_rows,
            args.cache,
            args.row_index,
            args.rechunk,
            args.cloud_options,
        )?
        .build()
        .into();
        lf.opt_state.file_caching = true;

        Ok(lf)
    }

    fn path(&self) -> &Path {
        self.path.as_path()
    }

    fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    fn with_path(mut self, path: PathBuf) -> Self {
        self.path = resolve_homedir(&path);
        self
    }

    fn with_paths(mut self, paths: Arc<[PathBuf]>) -> Self {
        let paths = paths
            .iter()
            .map(|p| resolve_homedir(p))
            .collect::<Vec<_>>()
            .into();

        self.paths = paths;
        self
    }

    fn with_n_rows(mut self, n_rows: impl Into<Option<usize>>) -> Self {
        self.args.n_rows = n_rows.into();
        self
    }

    fn with_row_index(mut self, row_index: impl Into<Option<RowIndex>>) -> Self {
        self.args.row_index = row_index.into();
        self
    }

    fn rechunk(&self) -> bool {
        self.args.rechunk
    }

    fn with_rechunk(mut self, toggle: bool) -> Self {
        self.args.rechunk = toggle;
        self
    }

    fn n_rows(&self) -> Option<usize> {
        self.args.n_rows
    }

    fn row_index(&self) -> Option<&RowIndex> {
        self.args.row_index.as_ref()
    }
}

impl LazyFrame {
    /// Create a LazyFrame directly from a ipc scan.
    pub fn scan_ipc(path: impl AsRef<Path>, args: ScanArgsIpc) -> PolarsResult<Self> {
        LazyIpcReader::new(path.as_ref().to_owned(), args).finish()
    }

    pub fn scan_ipc_files(paths: Arc<[PathBuf]>, args: ScanArgsIpc) -> PolarsResult<Self> {
        LazyIpcReader::new(PathBuf::new(), args)
            .with_paths(paths)
            .finish()
    }
}
