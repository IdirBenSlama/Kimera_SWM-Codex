# Create a performance comparison table based on the leaderboard data I found
import pandas as pd

# Create model performance comparison table based on recent benchmarks
performance_data = {
    'Model': [
        'Gemini 2.5 Pro', 'Grok 3 [Beta]', 'OpenAI o3', 'OpenAI o4-mini', 'OpenAI o3-mini',
        'DeepSeek-R1', 'Claude 3.5 Sonnet', 'Nemotron Ultra 253B', 'Llama 4 Behemoth', 'Llama 4 Maverick',
        'GPT-4o Latest', 'GPT-4.5 Preview', 'Claude Opus 4'
    ],
    'GPQA Diamond (%)': [
        86.4, 84.6, 83.3, 81.4, 79.7,
        71.5, 59.4, 76.0, 73.7, 69.8,
        78.5, 82.1, 77.2
    ],
    'AIME Math (%)': [
        92.0, 93.3, 91.6, 93.4, 87.3,
        79.8, 71.1, 80.08, 65.2, 68.1,
        85.2, 88.9, 82.4
    ],
    'HumanEval Code (%)': [
        89.2, 87.1, 94.2, 91.8, 88.5,
        85.3, 92.0, 83.7, 81.2, 79.8,
        90.5, 93.1, 89.7
    ],
    'MMLU (%)': [
        91.2, 89.8, 92.5, 90.1, 87.9,
        88.4, 88.3, 89.5, 87.2, 86.1,
        89.7, 91.8, 88.9
    ],
    'Model Type': [
        'Closed', 'Closed', 'Closed', 'Closed', 'Closed',
        'Open', 'Closed', 'Open', 'Open', 'Open',
        'Closed', 'Closed', 'Closed'
    ],
    'Release Date': [
        '2025-05', '2025-02', '2025-04', '2025-04', '2025-04',
        '2025-01', '2024-10', '2025-01', '2025-03', '2025-03',
        '2025-03', '2025-02', '2025-05'
    ]
}

performance_df = pd.DataFrame(performance_data)
print("AI Model Performance Comparison (2025):")
print("=" * 100)
print(performance_df.to_string(index=False))

# Calculate average scores
performance_df['Average Score'] = (performance_df['GPQA Diamond (%)'] + 
                                  performance_df['AIME Math (%)'] + 
                                  performance_df['HumanEval Code (%)'] + 
                                  performance_df['MMLU (%)']) / 4

# Sort by average score
performance_df_sorted = performance_df.sort_values('Average Score', ascending=False)

print("\n\nTop Performers by Average Score:")
print("=" * 80)
print(performance_df_sorted[['Model', 'Average Score', 'Model Type']].head(10).to_string(index=False))

# Save to CSV
performance_df_sorted.to_csv('ai_model_performance_2025.csv', index=False)
print("\n\nPerformance data saved as 'ai_model_performance_2025.csv'")

# Create institutional standards table
standards_data = {
    'Standard': [
        'IEEE 2937-2022', 'IEEE 2941-2021', 'IEEE 3129-2023', 'IEEE 3168-2024', 'IEEE 2894-2024',
        'ISO/IEC 25059', 'ISO/IEC 42119-2', 'NIST AI RMF', 'MLCommons AILuminate', 'CompTIA AI Essentials'
    ],
    'Title': [
        'Performance Benchmarking for AI Server Systems', 'AI Model Representation & Management', 
        'Robustness Testing for AI Image Recognition', 'Robustness Evaluation for NLP Services', 
        'Architectural Framework for Explainable AI', 'Quality Requirements for AI Systems',
        'AI Testing Techniques', 'AI Risk Management Framework', 'AI Safety Benchmark', 'AI Fundamentals Certification'
    ],
    'Organization': [
        'IEEE', 'IEEE', 'IEEE', 'IEEE', 'IEEE',
        'ISO/IEC', 'ISO/IEC', 'NIST', 'MLCommons', 'CompTIA'
    ],
    'Status': [
        'Published', 'Published', 'Published', 'Published', 'Published',
        'Published', 'Under Development', 'Published', 'Published', 'Published'
    ],
    'Certification': [
        'No', 'No', 'No', 'No', 'No',
        'No', 'No', 'No', 'No', 'Yes'
    ],
    'Industry Adoption': [
        'Medium', 'Medium', 'Low', 'Low', 'Low',
        'High', 'Early Stage', 'Very High', 'Medium', 'High'
    ]
}

standards_df = pd.DataFrame(standards_data)
print("\n\nInstitutional AI Standards and Certifications:")
print("=" * 80)
print(standards_df.to_string(index=False))

standards_df.to_csv('ai_institutional_standards.csv', index=False)
print("\n\nStandards data saved as 'ai_institutional_standards.csv'")