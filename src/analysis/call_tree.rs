use std::collections::HashMap;

use crate::profile::resolved::ResolvedThread;

#[derive(Debug, Clone)]
pub struct CallTreeNode {
    pub function_name: String,
    pub self_time_ms: f64,
    pub total_time_ms: f64,
    pub total_percent: f64,
    pub children: Vec<CallTreeNode>,
}

pub fn build_call_tree(
    thread: &ResolvedThread,
    max_depth: usize,
    min_percent: f64,
) -> CallTreeNode {
    let interval_ms = if thread.samples.len() >= 2 {
        thread.duration_ms / (thread.samples.len() - 1) as f64
    } else {
        1.0
    };

    let total_time = interval_ms * thread.samples.len() as f64;

    let mut root = TreeBuilder {
        name: "(root)".to_string(),
        self_time: 0.0,
        total_time: 0.0,
        children: HashMap::new(),
    };

    for sample in &thread.samples {
        if sample.stack.is_empty() {
            root.self_time += interval_ms;
            root.total_time += interval_ms;
            continue;
        }

        root.total_time += interval_ms;
        let mut node = &mut root;

        let stack_len = sample.stack.len();
        for (depth, frame) in sample.stack.iter().enumerate() {
            if depth >= max_depth {
                // Stack is deeper than max_depth — stop descending but do NOT
                // attribute this sample as self-time of the last visible node,
                // since that node is not actually a leaf; the time is just hidden.
                break;
            }

            node = node
                .children
                .entry(frame.function_name.clone())
                .or_insert_with(|| TreeBuilder {
                    name: frame.function_name.clone(),
                    self_time: 0.0,
                    total_time: 0.0,
                    children: HashMap::new(),
                });
            node.total_time += interval_ms;

            // Attribute self-time only to the true leaf frame of the sample
            if depth == stack_len - 1 {
                node.self_time += interval_ms;
            }
        }
    }

    root.into_node(total_time, min_percent)
}

struct TreeBuilder {
    name: String,
    self_time: f64,
    total_time: f64,
    children: HashMap<String, TreeBuilder>,
}

impl TreeBuilder {
    fn into_node(self, total_profile_time: f64, min_percent: f64) -> CallTreeNode {
        let total_percent = if total_profile_time > 0.0 {
            self.total_time / total_profile_time * 100.0
        } else {
            0.0
        };

        let mut children: Vec<CallTreeNode> = self
            .children
            .into_values()
            .filter(|c| {
                let pct = if total_profile_time > 0.0 {
                    c.total_time / total_profile_time * 100.0
                } else {
                    0.0
                };
                pct >= min_percent
            })
            .map(|c| c.into_node(total_profile_time, min_percent))
            .collect();

        children.sort_by(|a, b| b.total_time_ms.partial_cmp(&a.total_time_ms).unwrap());

        CallTreeNode {
            function_name: self.name,
            self_time_ms: self.self_time,
            total_time_ms: self.total_time,
            total_percent,
            children,
        }
    }
}

impl CallTreeNode {
    pub fn render_text(&self, max_depth: usize) -> String {
        let mut output = String::new();
        self.render_recursive(&mut output, 0, max_depth);
        output
    }

    fn render_recursive(&self, output: &mut String, depth: usize, max_depth: usize) {
        if depth > max_depth {
            return;
        }
        let indent = "  ".repeat(depth);
        output.push_str(&format!(
            "{}{} [{:.1}% total, {:.1}ms self]\n",
            indent, self.function_name, self.total_percent, self.self_time_ms,
        ));
        for child in &self.children {
            child.render_recursive(output, depth + 1, max_depth);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::resolved::{ResolvedFrame, ResolvedSample};

    fn make_frame(name: &str) -> ResolvedFrame {
        ResolvedFrame {
            function_name: name.to_string(),
            file: None,
            line: None,
            category: "Other".to_string(),
            library: None,
        }
    }

    #[test]
    fn test_call_tree_basic() {
        let thread = ResolvedThread {
            name: "test".to_string(),
            pid: "1".to_string(),
            tid: "1".to_string(),
            is_main: true,
            duration_ms: 20.0,
            markers: vec![],
            samples: vec![
                ResolvedSample {
                    timestamp_ms: 0.0,
                    weight: 1,
                    cpu_delta_us: None,
                    stack: vec![make_frame("main"), make_frame("foo"), make_frame("bar")],
                },
                ResolvedSample {
                    timestamp_ms: 10.0,
                    weight: 1,
                    cpu_delta_us: None,
                    stack: vec![make_frame("main"), make_frame("foo")],
                },
                ResolvedSample {
                    timestamp_ms: 20.0,
                    weight: 1,
                    cpu_delta_us: None,
                    stack: vec![make_frame("main"), make_frame("baz")],
                },
            ],
        };

        let tree = build_call_tree(&thread, 10, 0.0);
        assert_eq!(tree.function_name, "(root)");
        assert_eq!(tree.children.len(), 1); // just "main"
        let main_node = &tree.children[0];
        assert_eq!(main_node.function_name, "main");
        assert_eq!(main_node.children.len(), 2); // "foo" and "baz"
    }
}
