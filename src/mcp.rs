use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, ServiceExt, schemars, tool, tool_handler, tool_router};
use serde::{Deserialize, Serialize};

use crate::analysis::call_tree;
use crate::analysis::flamegraph;
use crate::analysis::functions::{self, FunctionStats};
use crate::profile::parse::load_profile;
use crate::profile::resolved::{ResolvedProfile, ResolvedThread};

/// Pre-computed analysis data cached per thread
#[derive(Debug)]
struct AnalysisCache {
    function_stats: HashMap<String, Vec<FunctionStats>>,
}

impl AnalysisCache {
    fn build(profile: &ResolvedProfile) -> Self {
        let mut function_stats = HashMap::new();
        for thread in &profile.threads {
            function_stats.insert(
                thread.name.clone(),
                functions::compute_function_stats(thread),
            );
        }
        AnalysisCache { function_stats }
    }
}

/// A loaded and analyzed profile with its cache.
struct CachedProfile {
    profile: Arc<ResolvedProfile>,
    cache: Arc<AnalysisCache>,
}

#[derive(Clone)]
pub struct ProfileServer {
    profiles: Arc<Mutex<HashMap<PathBuf, Arc<CachedProfile>>>>,
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

impl std::fmt::Debug for ProfileServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProfileServer").finish()
    }
}

impl ProfileServer {
    pub fn new() -> Self {
        Self {
            profiles: Arc::new(Mutex::new(HashMap::new())),
            tool_router: Self::tool_router(),
        }
    }

    fn get_profile(&self, path: &str) -> Result<Arc<CachedProfile>, String> {
        let canonical =
            std::fs::canonicalize(path).map_err(|e| format!("Invalid path '{}': {}", path, e))?;

        {
            let cache = self.profiles.lock().unwrap();
            if let Some(cached) = cache.get(&canonical) {
                return Ok(Arc::clone(cached));
            }
        }

        let profile = load_profile(&canonical)
            .map_err(|e| format!("Failed to load profile '{}': {}", path, e))?;
        let analysis_cache = AnalysisCache::build(&profile);
        let cached = Arc::new(CachedProfile {
            profile: Arc::new(profile),
            cache: Arc::new(analysis_cache),
        });

        self.profiles
            .lock()
            .unwrap()
            .insert(canonical, Arc::clone(&cached));
        Ok(cached)
    }
}

fn find_thread<'a>(
    profile: &'a ResolvedProfile,
    thread_name: &Option<String>,
) -> Option<&'a ResolvedThread> {
    match thread_name {
        Some(name) => {
            // Try exact match first, then case-insensitive substring
            profile
                .threads
                .iter()
                .find(|t| t.name == *name)
                .or_else(|| {
                    let lower = name.to_lowercase();
                    profile
                        .threads
                        .iter()
                        .find(|t| t.name.to_lowercase().contains(&lower))
                })
        }
        None => {
            // Default: prefer the main thread that actually has samples (avoids picking
            // the samply process itself which appears as an is_main thread with 0 samples)
            profile
                .threads
                .iter()
                .find(|t| t.is_main && !t.samples.is_empty())
                .or_else(|| profile.threads.iter().find(|t| !t.samples.is_empty()))
                .or_else(|| profile.threads.iter().find(|t| t.is_main))
                .or(profile.threads.first())
        }
    }
}

// ── Request types ──

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ProfileInfoRequest {
    #[schemars(description = "Path to the profile JSON file (or .json.gz)")]
    pub path: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ProfileThreadsRequest {
    #[schemars(description = "Path to the profile JSON file (or .json.gz)")]
    pub path: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TopFunctionsRequest {
    #[schemars(description = "Path to the profile JSON file (or .json.gz)")]
    pub path: String,

    #[schemars(description = "Thread name to analyze (omit for main thread with most samples). Use profile_threads to list available thread names.")]
    #[serde(default)]
    pub thread: Option<String>,

    #[schemars(description = "Sort by 'self' / 'self_time' (default, CPU hotspots) or 'total' / 'total_time' (dominates call tree)")]
    #[serde(default = "default_sort_by")]
    pub sort_by: String,

    #[schemars(description = "Maximum number of functions to return (default: 20)")]
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_sort_by() -> String {
    "self".to_string()
}

fn default_limit() -> usize {
    20
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CallTreeRequest {
    #[schemars(description = "Path to the profile JSON file (or .json.gz)")]
    pub path: String,

    #[schemars(description = "Thread name to analyze (omit for main thread with most samples). Use profile_threads to list names.")]
    #[serde(default)]
    pub thread: Option<String>,

    #[schemars(description = "Maximum depth of the call tree (default: 10)")]
    #[serde(default = "default_max_depth")]
    pub max_depth: usize,

    #[schemars(description = "Minimum percentage to include a node (default: 1.0)")]
    #[serde(default = "default_min_percent")]
    pub min_percent: f64,
}

fn default_max_depth() -> usize {
    10
}

fn default_min_percent() -> f64 {
    1.0
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FunctionDetailRequest {
    #[schemars(description = "Path to the profile JSON file (or .json.gz)")]
    pub path: String,

    #[schemars(description = "Name of the function to look up")]
    pub function_name: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct MarkersRequest {
    #[schemars(description = "Path to the profile JSON file (or .json.gz)")]
    pub path: String,

    #[schemars(description = "Thread name to analyze (omit for main thread with most samples). Use profile_threads to list names.")]
    #[serde(default)]
    pub thread: Option<String>,

    #[schemars(description = "Maximum number of markers to return (default: 50)")]
    #[serde(default = "default_marker_limit")]
    pub limit: usize,
}

fn default_marker_limit() -> usize {
    50
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FlamegraphRequest {
    #[schemars(description = "Path to the profile JSON file (or .json.gz)")]
    pub path: String,

    #[schemars(description = "Thread name to analyze (omit for main thread with most samples). Use profile_threads to list names.")]
    #[serde(default)]
    pub thread: Option<String>,
}

// ── Response types ──

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ProfileInfoResult {
    pub product: String,
    pub duration_ms: f64,
    pub total_samples: usize,
    pub thread_count: usize,
    pub interval_ms: f64,
    pub categories: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ThreadInfo {
    pub name: String,
    pub tid: String,
    pub sample_count: usize,
    pub duration_ms: f64,
    pub is_main: bool,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ProfileThreadsResult {
    pub threads: Vec<ThreadInfo>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct FunctionRow {
    pub name: String,
    pub self_percent: f64,
    pub total_percent: f64,
    pub self_time_ms: f64,
    pub total_time_ms: f64,
    pub sample_count: usize,
    pub library: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TopFunctionsResult {
    pub thread: String,
    pub sort_by: String,
    pub functions: Vec<FunctionRow>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CallerCalleeEntry {
    pub function_name: String,
    pub count: usize,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct FunctionDetailResult {
    pub function_name: String,
    pub self_time_ms: f64,
    pub total_time_ms: f64,
    pub self_percent: f64,
    pub total_percent: f64,
    pub sample_count: usize,
    pub library: Option<String>,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub callers: Vec<CallerCalleeEntry>,
    pub callees: Vec<CallerCalleeEntry>,
    pub threads: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CallTreeResult {
    pub thread: String,
    pub tree: String,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct MarkerInfo {
    pub name: String,
    pub start_time: Option<f64>,
    pub end_time: Option<f64>,
    pub category: String,
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct MarkersResult {
    pub thread: String,
    pub markers: Vec<MarkerInfo>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct FlamegraphResult {
    pub thread: String,
    pub collapsed_stacks: String,
}

// ── Tool implementations ──

#[tool_router]
impl ProfileServer {
    #[tool(
        description = "Get profile metadata: duration, sample count, thread count, sampling interval, and categories"
    )]
    fn profile_info(
        &self,
        Parameters(req): Parameters<ProfileInfoRequest>,
    ) -> Result<Json<ProfileInfoResult>, String> {
        let cached = self.get_profile(&req.path)?;
        let profile = &cached.profile;

        Ok(Json(ProfileInfoResult {
            product: profile.product.clone(),
            duration_ms: profile.duration_ms,
            total_samples: profile.total_sample_count,
            thread_count: profile.threads.len(),
            interval_ms: profile.interval_ms,
            categories: profile.categories.clone(),
        }))
    }

    #[tool(
        description = "List all threads with names, sample counts, time ranges, and whether they are the main thread"
    )]
    fn profile_threads(
        &self,
        Parameters(req): Parameters<ProfileThreadsRequest>,
    ) -> Result<Json<ProfileThreadsResult>, String> {
        let cached = self.get_profile(&req.path)?;
        let profile = &cached.profile;

        let threads = profile
            .threads
            .iter()
            .map(|t| ThreadInfo {
                name: t.name.clone(),
                tid: t.tid.clone(),
                sample_count: t.samples.len(),
                duration_ms: t.duration_ms,
                is_main: t.is_main,
            })
            .collect();

        Ok(Json(ProfileThreadsResult { threads }))
    }

    #[tool(
        description = "Get top N functions by self-time or total-time for a thread. Use sort_by='self' for CPU hotspots, sort_by='total' for functions dominating the call tree."
    )]
    fn profile_top_functions(
        &self,
        Parameters(req): Parameters<TopFunctionsRequest>,
    ) -> Result<Json<TopFunctionsResult>, String> {
        let cached = self.get_profile(&req.path)?;
        let thread = find_thread(&cached.profile, &req.thread).ok_or("Thread not found")?;

        let mut stats = cached
            .cache
            .function_stats
            .get(&thread.name)
            .cloned()
            .unwrap_or_default();

        // Accept "total" or "total_time"; anything else defaults to self-time
        let sort_by_total = matches!(req.sort_by.as_str(), "total" | "total_time");
        if sort_by_total {
            stats.sort_by(|a, b| b.total_time_ms.partial_cmp(&a.total_time_ms).unwrap());
        }
        // "self" / "self_time" / anything else → already sorted by self-time from cache

        let functions = stats
            .into_iter()
            .take(req.limit)
            .map(|s| FunctionRow {
                name: s.name,
                self_percent: round2(s.self_percent),
                total_percent: round2(s.total_percent),
                self_time_ms: round2(s.self_time_ms),
                total_time_ms: round2(s.total_time_ms),
                sample_count: s.sample_count,
                library: s.library,
            })
            .collect();

        Ok(Json(TopFunctionsResult {
            thread: thread.name.clone(),
            sort_by: req.sort_by,
            functions,
        }))
    }

    #[tool(
        description = "Get detailed information about a specific function: callers, callees, source locations, time breakdown. Searches across all threads."
    )]
    fn profile_function_detail(
        &self,
        Parameters(req): Parameters<FunctionDetailRequest>,
    ) -> Result<Json<FunctionDetailResult>, String> {
        let cached = self.get_profile(&req.path)?;
        let profile = &cached.profile;
        let target = &req.function_name;

        let mut total_self_time = 0.0;
        let mut total_total_time = 0.0;
        let mut total_self_pct = 0.0;
        let mut total_total_pct = 0.0;
        let mut total_samples = 0usize;
        let mut library = None;
        let mut file = None;
        let mut line = None;
        let mut callers: HashMap<String, usize> = HashMap::new();
        let mut callees: HashMap<String, usize> = HashMap::new();
        let mut found_threads = Vec::new();

        // Resolve to the actual full function name using exact match, then suffix, then substring
        let resolved_name: String = {
            let match_fn = |name: &str| -> bool {
                name == target.as_str()
                    || name.ends_with(&format!("::{target}"))
                    || name.contains(target.as_str())
            };
            profile
                .threads
                .iter()
                .find_map(|thread| {
                    cached
                        .cache
                        .function_stats
                        .get(&thread.name)?
                        .iter()
                        .find(|s| match_fn(&s.name))
                        .map(|s| s.name.clone())
                })
                .unwrap_or_else(|| target.clone())
        };
        let target = &resolved_name;

        for thread in &profile.threads {
            // Check function stats
            if let Some(stats) = cached.cache.function_stats.get(&thread.name)
                && let Some(s) = stats.iter().find(|s| s.name == *target)
            {
                total_self_time += s.self_time_ms;
                total_total_time += s.total_time_ms;
                total_self_pct += s.self_percent;
                total_total_pct += s.total_percent;
                total_samples += s.sample_count;
                if library.is_none() {
                    library = s.library.clone();
                }
                if file.is_none() {
                    file = s.file.clone();
                }
                if line.is_none() {
                    line = s.line;
                }
                found_threads.push(thread.name.clone());
            }

            // Find callers and callees from actual stacks
            for sample in &thread.samples {
                for (i, frame) in sample.stack.iter().enumerate() {
                    if frame.function_name == *target {
                        // Caller is the previous frame in the stack
                        if i > 0 {
                            *callers
                                .entry(sample.stack[i - 1].function_name.clone())
                                .or_default() += 1;
                        }
                        // Callee is the next frame
                        if i + 1 < sample.stack.len() {
                            *callees
                                .entry(sample.stack[i + 1].function_name.clone())
                                .or_default() += 1;
                        }
                    }
                }
            }
        }

        if found_threads.is_empty() {
            return Err(format!("Function '{}' not found in any thread", target));
        }

        let mut callers: Vec<CallerCalleeEntry> = callers
            .into_iter()
            .map(|(name, count)| CallerCalleeEntry {
                function_name: name,
                count,
            })
            .collect();
        callers.sort_by(|a, b| b.count.cmp(&a.count));

        let mut callees: Vec<CallerCalleeEntry> = callees
            .into_iter()
            .map(|(name, count)| CallerCalleeEntry {
                function_name: name,
                count,
            })
            .collect();
        callees.sort_by(|a, b| b.count.cmp(&a.count));

        Ok(Json(FunctionDetailResult {
            function_name: target.clone(),
            self_time_ms: round2(total_self_time),
            total_time_ms: round2(total_total_time),
            self_percent: round2(total_self_pct),
            total_percent: round2(total_total_pct),
            sample_count: total_samples,
            library,
            file,
            line,
            callers,
            callees,
            threads: found_threads,
        }))
    }

    #[tool(
        description = "Get a hierarchical call tree showing where time is spent, with configurable depth and minimum percentage threshold"
    )]
    fn profile_call_tree(
        &self,
        Parameters(req): Parameters<CallTreeRequest>,
    ) -> Result<Json<CallTreeResult>, String> {
        let cached = self.get_profile(&req.path)?;
        let thread = find_thread(&cached.profile, &req.thread).ok_or("Thread not found")?;

        let tree = call_tree::build_call_tree(thread, req.max_depth, req.min_percent);
        let text = tree.render_text(req.max_depth);

        Ok(Json(CallTreeResult {
            thread: thread.name.clone(),
            tree: text,
        }))
    }

    #[tool(
        description = "Get timeline markers/events for a thread, including their timing, category, and associated data"
    )]
    fn profile_markers(
        &self,
        Parameters(req): Parameters<MarkersRequest>,
    ) -> Result<Json<MarkersResult>, String> {
        let cached = self.get_profile(&req.path)?;
        let thread = find_thread(&cached.profile, &req.thread).ok_or("Thread not found")?;

        let markers: Vec<MarkerInfo> = thread
            .markers
            .iter()
            .take(req.limit)
            .map(|m| MarkerInfo {
                name: m.name.clone(),
                start_time: m.start_time,
                end_time: m.end_time,
                category: m.category.clone(),
                data: m.data.clone(),
            })
            .collect();

        Ok(Json(MarkersResult {
            thread: thread.name.clone(),
            markers,
        }))
    }

    #[tool(
        description = "Get collapsed stack format (Brendan Gregg's folded format) for generating flamegraphs. Each line: func1;func2;func3 count"
    )]
    fn profile_flamegraph(
        &self,
        Parameters(req): Parameters<FlamegraphRequest>,
    ) -> Result<Json<FlamegraphResult>, String> {
        let cached = self.get_profile(&req.path)?;
        let thread = find_thread(&cached.profile, &req.thread).ok_or("Thread not found")?;

        let stacks = flamegraph::collapsed_stacks(thread);

        Ok(Json(FlamegraphResult {
            thread: thread.name.clone(),
            collapsed_stacks: stacks,
        }))
    }
}

// ── Server handler ──

#[tool_handler]
impl ServerHandler for ProfileServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Samply profiler analysis tools. Every tool requires a `path` parameter \
                 pointing to a profile JSON file (or .json.gz). Use profile_info for metadata \
                 overview, profile_threads to list threads, profile_top_functions to find CPU \
                 hotspots, profile_call_tree for hierarchical call analysis, \
                 profile_function_detail for deep-dive into a specific function's \
                 callers/callees, profile_markers for timeline events, and profile_flamegraph \
                 for collapsed stack output."
                    .to_string(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

pub async fn run_server() -> Result<()> {
    let server = ProfileServer::new();

    let service = server
        .serve((tokio::io::stdin(), tokio::io::stdout()))
        .await?;

    service.waiting().await?;

    Ok(())
}
