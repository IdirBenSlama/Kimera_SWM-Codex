#!/usr/bin/env python3
"""
KIMERA SWM System - Documentation Generator
==========================================

Phase 3.3: Documentation & Knowledge Management Implementation
Provides comprehensive documentation generation, API documentation, and knowledge management.

Features:
- Automated API documentation generation
- Code documentation analysis
- Knowledge base management
- Documentation coverage reporting
- Interactive documentation generation
- Documentation quality metrics
- Cross-reference generation

Author: KIMERA Development Team
Date: 2025-01-31
Phase: 3.3 - Documentation & Knowledge Management
"""

import ast
import inspect
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import markdown
import jinja2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentationEntry:
    """Represents a documentation entry."""
    name: str
    doc_type: str  # 'module', 'class', 'function', 'method'
    docstring: Optional[str]
    file_path: str
    line_number: int
    signature: Optional[str] = None
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    see_also: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class DocumentationMetrics:
    """Documentation quality metrics."""
    total_items: int = 0
    documented_items: int = 0
    coverage_percentage: float = 0.0
    missing_docstrings: List[str] = field(default_factory=list)
    poor_quality_docs: List[str] = field(default_factory=list)
    average_docstring_length: float = 0.0
    completeness_score: float = 0.0

class DocstringParser:
    """Parser for extracting structured information from docstrings."""
    
    def __init__(self):
        self.section_patterns = {
            'parameters': r'(?:Args?|Parameters?):\s*\n((?:\s*\w+.*\n?)*)',
            'returns': r'(?:Returns?):\s*\n(.*?)(?=\n\s*\w+:|$)',
            'raises': r'(?:Raises?):\s*\n((?:\s*\w+.*\n?)*)',
            'examples': r'(?:Examples?):\s*\n(.*?)(?=\n\s*\w+:|$)',
            'notes': r'(?:Notes?):\s*\n(.*?)(?=\n\s*\w+:|$)',
            'see_also': r'(?:See Also):\s*\n(.*?)(?=\n\s*\w+:|$)',
            'warnings': r'(?:Warnings?):\s*\n(.*?)(?=\n\s*\w+:|$)'
        }
    
    def parse_docstring(self, docstring: str) -> Dict[str, Any]:
        """Parse structured docstring into components."""
        if not docstring:
            return {}
        
        result = {
            'summary': '',
            'description': '',
            'parameters': [],
            'returns': '',
            'raises': [],
            'examples': [],
            'notes': [],
            'see_also': [],
            'warnings': []
        }
        
        # Extract summary (first line)
        lines = docstring.strip().split('\n')
        if lines:
            result['summary'] = lines[0].strip()
        
        # Extract sections using regex
        for section, pattern in self.section_patterns.items():
            match = re.search(pattern, docstring, re.MULTILINE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                
                if section == 'parameters':
                    result[section] = self._parse_parameters(content)
                elif section in ['examples', 'notes', 'see_also', 'warnings']:
                    result[section] = [line.strip() for line in content.split('\n') if line.strip()]
                else:
                    result[section] = content
        
        # Extract description (everything before first section)
        desc_end = len(docstring)
        for pattern in self.section_patterns.values():
            match = re.search(pattern, docstring, re.MULTILINE)
            if match:
                desc_end = min(desc_end, match.start())
        
        description_text = docstring[:desc_end].strip()
        if description_text and len(lines) > 1:
            result['description'] = '\n'.join(lines[1:]).strip()
        
        return result
    
    def _parse_parameters(self, params_text: str) -> List[Dict[str, str]]:
        """Parse parameter section into structured format."""
        parameters = []
        
        for line in params_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Look for parameter pattern: name (type): description
            param_match = re.match(r'(\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.*)', line)
            if param_match:
                name, param_type, description = param_match.groups()
                parameters.append({
                    'name': name,
                    'type': param_type or 'Any',
                    'description': description
                })
        
        return parameters

class DocumentationExtractor:
    """Extracts documentation from Python source code."""
    
    def __init__(self, root_path: str = "src"):
        self.root_path = Path(root_path)
        self.parser = DocstringParser()
        self.documentation_entries: List[DocumentationEntry] = []
    
    def extract_documentation(self) -> List[DocumentationEntry]:
        """Extract documentation from all Python files."""
        logger.info("🔍 Extracting documentation from source code...")
        
        self.documentation_entries.clear()
        
        for py_file in self.root_path.rglob("*.py"):
            if self._should_process_file(py_file):
                self._extract_from_file(py_file)
        
        logger.info(f"Extracted documentation for {len(self.documentation_entries)} items")
        return self.documentation_entries
    
    def _should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed."""
        exclude_patterns = [
            "__pycache__",
            ".git",
            "migrations",
            "venv",
            ".venv",
            "test_",
            "_test"
        ]
        
        return not any(pattern in str(file_path) for pattern in exclude_patterns)
    
    def _extract_from_file(self, file_path: Path):
        """Extract documentation from a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract module docstring
            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                self.documentation_entries.append(DocumentationEntry(
                    name=str(file_path.relative_to(self.root_path)),
                    doc_type='module',
                    docstring=module_docstring,
                    file_path=str(file_path),
                    line_number=1
                ))
            
            # Walk the AST to find classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._extract_class_documentation(node, file_path)
                elif isinstance(node, ast.FunctionDef):
                    self._extract_function_documentation(node, file_path)
        
        except Exception as e:
            logger.warning(f"Error extracting documentation from {file_path}: {e}")
    
    def _extract_class_documentation(self, node: ast.ClassDef, file_path: Path):
        """Extract documentation for a class."""
        docstring = ast.get_docstring(node)
        parsed_doc = self.parser.parse_docstring(docstring) if docstring else {}
        
        entry = DocumentationEntry(
            name=node.name,
            doc_type='class',
            docstring=docstring,
            file_path=str(file_path),
            line_number=node.lineno,
            parameters=parsed_doc.get('parameters', []),
            examples=parsed_doc.get('examples', []),
            see_also=parsed_doc.get('see_also', []),
            notes=parsed_doc.get('notes', []),
            warnings=parsed_doc.get('warnings', [])
        )
        
        self.documentation_entries.append(entry)
        
        # Extract method documentation
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self._extract_method_documentation(item, file_path, node.name)
    
    def _extract_method_documentation(self, node: ast.FunctionDef, file_path: Path, class_name: str):
        """Extract documentation for a method."""
        docstring = ast.get_docstring(node)
        parsed_doc = self.parser.parse_docstring(docstring) if docstring else {}
        
        # Generate signature
        signature = self._generate_signature(node)
        
        entry = DocumentationEntry(
            name=f"{class_name}.{node.name}",
            doc_type='method',
            docstring=docstring,
            file_path=str(file_path),
            line_number=node.lineno,
            signature=signature,
            parameters=parsed_doc.get('parameters', []),
            return_type=parsed_doc.get('returns'),
            examples=parsed_doc.get('examples', []),
            see_also=parsed_doc.get('see_also', []),
            notes=parsed_doc.get('notes', []),
            warnings=parsed_doc.get('warnings', [])
        )
        
        self.documentation_entries.append(entry)
    
    def _extract_function_documentation(self, node: ast.FunctionDef, file_path: Path):
        """Extract documentation for a function."""
        docstring = ast.get_docstring(node)
        parsed_doc = self.parser.parse_docstring(docstring) if docstring else {}
        
        # Generate signature
        signature = self._generate_signature(node)
        
        entry = DocumentationEntry(
            name=node.name,
            doc_type='function',
            docstring=docstring,
            file_path=str(file_path),
            line_number=node.lineno,
            signature=signature,
            parameters=parsed_doc.get('parameters', []),
            return_type=parsed_doc.get('returns'),
            examples=parsed_doc.get('examples', []),
            see_also=parsed_doc.get('see_also', []),
            notes=parsed_doc.get('notes', []),
            warnings=parsed_doc.get('warnings', [])
        )
        
        self.documentation_entries.append(entry)
    
    def _generate_signature(self, node: ast.FunctionDef) -> str:
        """Generate function signature string."""
        args = []
        
        # Regular arguments
        for arg in node.args.args:
            args.append(arg.arg)
        
        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        
        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        return f"{node.name}({', '.join(args)})"

class DocumentationMetricsCalculator:
    """Calculates documentation quality metrics."""
    
    def __init__(self):
        self.quality_thresholds = {
            'min_docstring_length': 20,
            'good_docstring_length': 100,
            'required_sections': ['summary', 'parameters', 'returns']
        }
    
    def calculate_metrics(self, entries: List[DocumentationEntry]) -> DocumentationMetrics:
        """Calculate comprehensive documentation metrics."""
        logger.info("📊 Calculating documentation metrics...")
        
        metrics = DocumentationMetrics()
        
        total_items = len(entries)
        documented_items = len([e for e in entries if e.docstring])
        
        metrics.total_items = total_items
        metrics.documented_items = documented_items
        metrics.coverage_percentage = (documented_items / total_items * 100) if total_items > 0 else 0
        
        # Find missing docstrings
        metrics.missing_docstrings = [
            f"{e.name} ({e.doc_type}) in {e.file_path}:{e.line_number}"
            for e in entries if not e.docstring
        ]
        
        # Find poor quality documentation
        metrics.poor_quality_docs = []
        docstring_lengths = []
        
        for entry in entries:
            if entry.docstring:
                length = len(entry.docstring)
                docstring_lengths.append(length)
                
                if length < self.quality_thresholds['min_docstring_length']:
                    metrics.poor_quality_docs.append(
                        f"{entry.name} ({entry.doc_type}) - too short ({length} chars)"
                    )
        
        # Calculate average docstring length
        if docstring_lengths:
            metrics.average_docstring_length = sum(docstring_lengths) / len(docstring_lengths)
        
        # Calculate completeness score
        metrics.completeness_score = self._calculate_completeness_score(entries)
        
        return metrics
    
    def _calculate_completeness_score(self, entries: List[DocumentationEntry]) -> float:
        """Calculate documentation completeness score."""
        if not entries:
            return 0.0
        
        total_score = 0
        max_possible_score = 0
        
        for entry in entries:
            score = 0
            max_score = 10  # Base score for having documentation
            
            if entry.docstring:
                score += 5  # Base points for having docstring
                
                # Points for docstring quality
                if len(entry.docstring) >= self.quality_thresholds['min_docstring_length']:
                    score += 2
                
                if len(entry.docstring) >= self.quality_thresholds['good_docstring_length']:
                    score += 1
                
                # Points for having parameters documented
                if entry.parameters:
                    score += 1
                
                # Points for return type
                if entry.return_type:
                    score += 1
                
                # Points for examples
                if entry.examples:
                    score += 1
            
            total_score += score
            max_possible_score += max_score
        
        return (total_score / max_possible_score * 100) if max_possible_score > 0 else 0.0

class HTMLDocumentationGenerator:
    """Generates HTML documentation."""
    
    def __init__(self):
        self.jinja_env = jinja2.Environment(
            loader=jinja2.DictLoader(self._get_templates())
        )
    
    def generate_html_documentation(
        self, 
        entries: List[DocumentationEntry], 
        metrics: DocumentationMetrics,
        output_dir: str = "docs/html"
    ):
        """Generate complete HTML documentation."""
        logger.info("📄 Generating HTML documentation...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Group entries by type and module
        grouped_entries = self._group_entries(entries)
        
        # Generate index page
        self._generate_index_page(grouped_entries, metrics, output_path)
        
        # Generate individual module pages
        for module, module_entries in grouped_entries.items():
            self._generate_module_page(module, module_entries, output_path)
        
        # Generate metrics page
        self._generate_metrics_page(metrics, output_path)
        
        # Copy CSS and JS files
        self._copy_static_files(output_path)
        
        logger.info(f"HTML documentation generated in {output_path}")
    
    def _group_entries(self, entries: List[DocumentationEntry]) -> Dict[str, List[DocumentationEntry]]:
        """Group entries by module."""
        grouped = defaultdict(list)
        
        for entry in entries:
            if entry.doc_type == 'module':
                module_name = entry.name
            else:
                # Extract module name from file path
                file_path = Path(entry.file_path)
                parts = file_path.parts
                if 'src' in parts:
                    src_index = parts.index('src')
                    module_parts = parts[src_index + 1:]
                    module_name = '.'.join(module_parts[:-1]) + '.' + file_path.stem
                else:
                    module_name = file_path.stem
            
            grouped[module_name].append(entry)
        
        return dict(grouped)
    
    def _generate_index_page(
        self, 
        grouped_entries: Dict[str, List[DocumentationEntry]], 
        metrics: DocumentationMetrics,
        output_path: Path
    ):
        """Generate main index page."""
        template = self.jinja_env.get_template('index.html')
        
        content = template.render(
            title="KIMERA SWM System Documentation",
            modules=list(grouped_entries.keys()),
            metrics=metrics,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        with open(output_path / "index.html", 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_module_page(
        self, 
        module_name: str, 
        entries: List[DocumentationEntry], 
        output_path: Path
    ):
        """Generate page for a specific module."""
        template = self.jinja_env.get_template('module.html')
        
        # Separate entries by type
        classes = [e for e in entries if e.doc_type == 'class']
        functions = [e for e in entries if e.doc_type == 'function']
        methods = [e for e in entries if e.doc_type == 'method']
        
        content = template.render(
            module_name=module_name,
            classes=classes,
            functions=functions,
            methods=methods,
            all_entries=entries
        )
        
        # Create safe filename
        safe_filename = module_name.replace('.', '_').replace('/', '_') + '.html'
        with open(output_path / safe_filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_metrics_page(self, metrics: DocumentationMetrics, output_path: Path):
        """Generate documentation metrics page."""
        template = self.jinja_env.get_template('metrics.html')
        
        content = template.render(
            metrics=metrics,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        with open(output_path / "metrics.html", 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _copy_static_files(self, output_path: Path):
        """Copy CSS and JavaScript files."""
        static_dir = output_path / "static"
        static_dir.mkdir(exist_ok=True)
        
        # Generate CSS
        css_content = self._get_css_content()
        with open(static_dir / "styles.css", 'w') as f:
            f.write(css_content)
        
        # Generate JavaScript
        js_content = self._get_js_content()
        with open(static_dir / "scripts.js", 'w') as f:
            f.write(js_content)
    
    def _get_templates(self) -> Dict[str, str]:
        """Get Jinja2 templates."""
        return {
            'index.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link rel="stylesheet" href="static/styles.css">
</head>
<body>
    <header>
        <h1>{{ title }}</h1>
        <nav>
            <a href="index.html">Home</a>
            <a href="metrics.html">Metrics</a>
        </nav>
    </header>
    
    <main>
        <section class="overview">
            <h2>Documentation Overview</h2>
            <div class="metrics-summary">
                <div class="metric">
                    <span class="value">{{ metrics.coverage_percentage|round(1) }}%</span>
                    <span class="label">Coverage</span>
                </div>
                <div class="metric">
                    <span class="value">{{ metrics.documented_items }}</span>
                    <span class="label">Documented</span>
                </div>
                <div class="metric">
                    <span class="value">{{ metrics.total_items }}</span>
                    <span class="label">Total Items</span>
                </div>
                <div class="metric">
                    <span class="value">{{ metrics.completeness_score|round(1) }}%</span>
                    <span class="label">Quality Score</span>
                </div>
            </div>
        </section>
        
        <section class="modules">
            <h2>Modules</h2>
            <div class="module-grid">
                {% for module in modules %}
                <div class="module-card">
                    <h3><a href="{{ module.replace('.', '_').replace('/', '_') }}.html">{{ module }}</a></h3>
                </div>
                {% endfor %}
            </div>
        </section>
    </main>
    
    <footer>
        <p>Generated at {{ generated_at }}</p>
    </footer>
    
    <script src="static/scripts.js"></script>
</body>
</html>
            ''',
            
            'module.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ module_name }} - KIMERA Documentation</title>
    <link rel="stylesheet" href="static/styles.css">
</head>
<body>
    <header>
        <h1>{{ module_name }}</h1>
        <nav>
            <a href="index.html">Home</a>
            <a href="metrics.html">Metrics</a>
        </nav>
    </header>
    
    <main>
        {% if classes %}
        <section class="classes">
            <h2>Classes</h2>
            {% for class in classes %}
            <div class="doc-entry">
                <h3>{{ class.name }}</h3>
                {% if class.docstring %}
                <div class="docstring">{{ class.docstring|nl2br }}</div>
                {% endif %}
                <div class="meta">
                    <span class="file">{{ class.file_path }}:{{ class.line_number }}</span>
                </div>
            </div>
            {% endfor %}
        </section>
        {% endif %}
        
        {% if functions %}
        <section class="functions">
            <h2>Functions</h2>
            {% for function in functions %}
            <div class="doc-entry">
                <h3>{{ function.name }}</h3>
                {% if function.signature %}
                <div class="signature"><code>{{ function.signature }}</code></div>
                {% endif %}
                {% if function.docstring %}
                <div class="docstring">{{ function.docstring|nl2br }}</div>
                {% endif %}
                {% if function.parameters %}
                <div class="parameters">
                    <h4>Parameters:</h4>
                    <ul>
                    {% for param in function.parameters %}
                    <li><strong>{{ param.name }}</strong> ({{ param.type }}): {{ param.description }}</li>
                    {% endfor %}
                    </ul>
                </div>
                {% endif %}
                <div class="meta">
                    <span class="file">{{ function.file_path }}:{{ function.line_number }}</span>
                </div>
            </div>
            {% endfor %}
        </section>
        {% endif %}
    </main>
    
    <script src="static/scripts.js"></script>
</body>
</html>
            ''',
            
            'metrics.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documentation Metrics - KIMERA</title>
    <link rel="stylesheet" href="static/styles.css">
</head>
<body>
    <header>
        <h1>Documentation Metrics</h1>
        <nav>
            <a href="index.html">Home</a>
            <a href="metrics.html">Metrics</a>
        </nav>
    </header>
    
    <main>
        <section class="metrics-detail">
            <h2>Coverage Statistics</h2>
            <div class="stats-grid">
                <div class="stat">
                    <label>Total Items:</label>
                    <value>{{ metrics.total_items }}</value>
                </div>
                <div class="stat">
                    <label>Documented Items:</label>
                    <value>{{ metrics.documented_items }}</value>
                </div>
                <div class="stat">
                    <label>Coverage Percentage:</label>
                    <value>{{ metrics.coverage_percentage|round(1) }}%</value>
                </div>
                <div class="stat">
                    <label>Completeness Score:</label>
                    <value>{{ metrics.completeness_score|round(1) }}%</value>
                </div>
                <div class="stat">
                    <label>Average Docstring Length:</label>
                    <value>{{ metrics.average_docstring_length|round(0) }} chars</value>
                </div>
            </div>
        </section>
        
        {% if metrics.missing_docstrings %}
        <section class="missing-docs">
            <h2>Missing Documentation ({{ metrics.missing_docstrings|length }})</h2>
            <ul>
            {% for item in metrics.missing_docstrings[:20] %}
            <li>{{ item }}</li>
            {% endfor %}
            {% if metrics.missing_docstrings|length > 20 %}
            <li>... and {{ metrics.missing_docstrings|length - 20 }} more</li>
            {% endif %}
            </ul>
        </section>
        {% endif %}
        
        {% if metrics.poor_quality_docs %}
        <section class="poor-quality">
            <h2>Poor Quality Documentation ({{ metrics.poor_quality_docs|length }})</h2>
            <ul>
            {% for item in metrics.poor_quality_docs[:20] %}
            <li>{{ item }}</li>
            {% endfor %}
            {% if metrics.poor_quality_docs|length > 20 %}
            <li>... and {{ metrics.poor_quality_docs|length - 20 }} more</li>
            {% endif %}
            </ul>
        </section>
        {% endif %}
    </main>
    
    <footer>
        <p>Generated at {{ generated_at }}</p>
    </footer>
</body>
</html>
            '''
        }
    
    def _get_css_content(self) -> str:
        """Get CSS content for documentation."""
        return '''
/* KIMERA Documentation Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f8f9fa;
}

header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem 2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

header h1 {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

nav a {
    color: white;
    text-decoration: none;
    margin-right: 1rem;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    transition: background-color 0.3s;
}

nav a:hover {
    background-color: rgba(255,255,255,0.2);
}

main {
    max-width: 1200px;
    margin: 2rem auto;
    padding: 0 2rem;
}

.overview {
    background: white;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}

.metrics-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.metric {
    text-align: center;
    padding: 1rem;
    background: #f8f9fa;
    border-radius: 8px;
}

.metric .value {
    display: block;
    font-size: 2rem;
    font-weight: bold;
    color: #667eea;
}

.metric .label {
    font-size: 0.9rem;
    color: #666;
}

.module-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.module-card {
    background: white;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    transition: transform 0.3s, box-shadow 0.3s;
}

.module-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}

.module-card h3 a {
    color: #667eea;
    text-decoration: none;
}

.doc-entry {
    background: white;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.doc-entry h3 {
    color: #333;
    margin-bottom: 0.5rem;
}

.signature {
    background: #f8f9fa;
    padding: 0.5rem;
    border-radius: 4px;
    margin: 0.5rem 0;
    font-family: 'Monaco', 'Consolas', monospace;
}

.docstring {
    margin: 1rem 0;
    line-height: 1.7;
}

.parameters {
    margin: 1rem 0;
}

.parameters h4 {
    margin-bottom: 0.5rem;
    color: #667eea;
}

.parameters ul {
    margin-left: 1rem;
}

.meta {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #eee;
    font-size: 0.9rem;
    color: #666;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.stat {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid #eee;
}

.stat label {
    font-weight: 500;
}

.stat value {
    color: #667eea;
    font-weight: bold;
}

footer {
    text-align: center;
    padding: 2rem;
    color: #666;
    border-top: 1px solid #eee;
    margin-top: 3rem;
}

/* Responsive */
@media (max-width: 768px) {
    main {
        padding: 0 1rem;
    }
    
    header {
        padding: 1rem;
    }
    
    .metrics-summary {
        grid-template-columns: repeat(2, 1fr);
    }
    
    .module-grid {
        grid-template-columns: 1fr;
    }
}
        '''
    
    def _get_js_content(self) -> str:
        """Get JavaScript content for documentation."""
        return '''
// KIMERA Documentation JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Add search functionality
    addSearchFunctionality();
    
    // Add copy buttons to code blocks
    addCopyButtons();
    
    // Add smooth scrolling
    addSmoothScrolling();
});

function addSearchFunctionality() {
    // Simple search implementation
    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.placeholder = 'Search documentation...';
    searchInput.className = 'search-input';
    
    const nav = document.querySelector('nav');
    if (nav) {
        nav.appendChild(searchInput);
        
        searchInput.addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            const entries = document.querySelectorAll('.doc-entry, .module-card');
            
            entries.forEach(entry => {
                const text = entry.textContent.toLowerCase();
                if (text.includes(searchTerm) || searchTerm === '') {
                    entry.style.display = '';
                } else {
                    entry.style.display = 'none';
                }
            });
        });
    }
}

function addCopyButtons() {
    const codeBlocks = document.querySelectorAll('.signature code, pre code');
    
    codeBlocks.forEach(block => {
        const button = document.createElement('button');
        button.textContent = 'Copy';
        button.className = 'copy-button';
        
        button.addEventListener('click', function() {
            navigator.clipboard.writeText(block.textContent).then(() => {
                button.textContent = 'Copied!';
                setTimeout(() => {
                    button.textContent = 'Copy';
                }, 2000);
            });
        });
        
        block.parentElement.style.position = 'relative';
        block.parentElement.appendChild(button);
    });
}

function addSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
}
        '''

class DocumentationGenerator:
    """Main documentation generation coordinator."""
    
    def __init__(self, root_path: str = "src"):
        self.root_path = root_path
        self.extractor = DocumentationExtractor(root_path)
        self.metrics_calculator = DocumentationMetricsCalculator()
        self.html_generator = HTMLDocumentationGenerator()
    
    def generate_complete_documentation(self, output_dir: str = "docs") -> Dict[str, Any]:
        """Generate complete documentation suite."""
        logger.info("🚀 Starting complete documentation generation...")
        
        # Extract documentation
        entries = self.extractor.extract_documentation()
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(entries)
        
        # Generate HTML documentation
        html_output_dir = os.path.join(output_dir, "html")
        self.html_generator.generate_html_documentation(entries, metrics, html_output_dir)
        
        # Generate JSON report
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_entries": len(entries),
            "metrics": {
                "total_items": metrics.total_items,
                "documented_items": metrics.documented_items,
                "coverage_percentage": metrics.coverage_percentage,
                "completeness_score": metrics.completeness_score,
                "average_docstring_length": metrics.average_docstring_length,
                "missing_count": len(metrics.missing_docstrings),
                "poor_quality_count": len(metrics.poor_quality_docs)
            },
            "entries_by_type": {
                "modules": len([e for e in entries if e.doc_type == 'module']),
                "classes": len([e for e in entries if e.doc_type == 'class']),
                "functions": len([e for e in entries if e.doc_type == 'function']),
                "methods": len([e for e in entries if e.doc_type == 'method'])
            }
        }
        
        # Save JSON report
        report_path = os.path.join(output_dir, "documentation_report.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("✅ Documentation generation completed!")
        logger.info(f"📄 HTML documentation: {html_output_dir}")
        logger.info(f"📊 Report: {report_path}")
        
        return report

def main():
    """Main function to run documentation generation."""
    print("📚 KIMERA Documentation Generator")
    print("=" * 60)
    print("Phase 3.3: Documentation & Knowledge Management")
    print()
    
    generator = DocumentationGenerator()
    
    try:
        # Generate complete documentation
        report = generator.generate_complete_documentation()
        
        # Print summary
        print("📊 Documentation Generation Results:")
        print(f"   Total items analyzed: {report['total_entries']}")
        print(f"   Documentation coverage: {report['metrics']['coverage_percentage']:.1f}%")
        print(f"   Quality score: {report['metrics']['completeness_score']:.1f}%")
        print(f"   Missing documentation: {report['metrics']['missing_count']} items")
        print()
        print("📄 Generated files:")
        print("   - docs/html/index.html (Main documentation)")
        print("   - docs/html/metrics.html (Quality metrics)")
        print("   - docs/documentation_report.json (Detailed report)")
        print()
        print("🎯 Next steps:")
        print("   1. Open docs/html/index.html in your browser")
        print("   2. Review missing documentation items")
        print("   3. Improve docstring quality based on metrics")
        print("   4. Set up documentation CI/CD pipeline")
        
    except Exception as e:
        print(f"❌ Error during documentation generation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 