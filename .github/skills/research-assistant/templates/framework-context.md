# Framework Context

> Last analyzed: {{date}}
> Git SHA: {{git_sha}}

## Framework Overview

**Name:** {{framework_name}}
**Purpose:** {{brief_description}}
**Language:** {{language}}

## Config System

### Config Format
{{config_format_description}}

### Key Config Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| {{param}} | {{type}} | {{description}} |

### Example Config
```yaml
# Example configuration
{{example_config}}
```

## Experiment Entry Points

### Primary Entry Point
**File:** {{entry_point_file}}
**Command:** `{{run_command}}`

### Execution Flow
1. {{step_1}}
2. {{step_2}}
3. {{step_3}}

## Key Abstractions

### {{abstraction_1_name}}
- **Location:** {{file_path}}
- **Purpose:** {{purpose}}
- **Extension points:** {{how_to_extend}}

### {{abstraction_2_name}}
- **Location:** {{file_path}}
- **Purpose:** {{purpose}}
- **Extension points:** {{how_to_extend}}

## MLflow Integration

### Tracking Configuration
- **Tracking URI:** {{tracking_uri}}
- **Experiment naming:** {{naming_convention}}
- **Auto-logging:** {{enabled_or_disabled}}

### Logged Metrics
| Metric | Description | Logged By |
|--------|-------------|-----------|
| {{metric}} | {{description}} | {{component}} |

### Tags Used
| Tag | Purpose |
|-----|---------|
| {{tag}} | {{purpose}} |

## Reproducibility

### Random Seeds
{{seed_handling}}

### Checkpointing
{{checkpoint_strategy}}

## Known Patterns

### What Works Well
- {{pattern_1}}
- {{pattern_2}}

### Common Pitfalls
- {{pitfall_1}}
- {{pitfall_2}}

## Project Configuration
- **MLflow location:** local (mlflow.db)
- **NotebookLM notebooks:** configured by user at project setup
