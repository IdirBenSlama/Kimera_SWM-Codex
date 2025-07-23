#!/usr/bin/env python3
"""
Script to fix the route ordering in insight_router.py
"""

import re

# Read the file
with open('backend/api/routers/insight_router.py', 'r') as f:
    content = f.read()

# Find the position of the get_insight function
get_insight_match = re.search(r'(@router\.get\("/insights/\{insight_id\}".*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)', content, re.DOTALL)
if get_insight_match:
    get_insight_func = get_insight_match.group(1)
    
    # Find the position of the status function
    status_match = re.search(r'(@router\.get\("/insights/status".*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)', content, re.DOTALL)
    if status_match:
        status_func = status_match.group(1)
        
        # Remove both functions from their current positions
        content_without_funcs = content.replace(get_insight_func, "")
        content_without_funcs = content_without_funcs.replace(status_func, "")
        
        # Find where to insert them (after the generate_insight function)
        insert_pos = content_without_funcs.find('@router.get("/insights", tags=["Insights"])')
        
        if insert_pos > 0:
            # Insert status function first, then get_insight function
            new_content = (
                content_without_funcs[:insert_pos] +
                status_func + "\n\n" +
                get_insight_func + "\n\n" +
                content_without_funcs[insert_pos:]
            )
            
            # Write the fixed content
            with open('backend/api/routers/insight_router.py', 'w') as f:
                f.write(new_content)
            
            print("Fixed route ordering in insight_router.py")
        else:
            print("Could not find insertion point")
    else:
        print("Could not find status function")
else:
    print("Could not find get_insight function")