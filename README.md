# Large-text-AI-analysis-for-the-scoping-review-process
Systematic Review Fraud Detector

## Overview
AI-powered validation tool for detecting fabrication in student systematic review assignments

## Key Features
- 🔍 **Automated PubMed/ClinicalTrials.gov Search Reproduction**
- 📊 **Match Score Analysis** (80%+ pass, 50-80% warning, <50% fail)
- 🤖 **Dual-LLM Cross-Validation** with confidence scoring
- 📈 **Excel Reports** with color-coded quality indicators
- 🔗 **Evidence Traceability** with hyperlinks to source papers
- 🎓 **Educational Focus** with non-technical language for educators

- **6-Module Pipeline:**
1. Document Parser (Gemini 2.5 Flash)
2. Programmatic Search (Biopython + PubMed API)
3. Search Validator (Python set operations)
4. Paper Downloader (PMC API)
5. Dual-LLM Validator (Gemini + DeepSeek R1)
6. Report Generator (pandas + openpyxl)

🎯 **What Problem Does This Solve**?
The Problem:
Medical students are assigned to write systematic reviews as coursework. Some students cheat by:

**Faking the literature search** - Manually picking papers, then inventing a search strategy claiming those papers came from PubMed
**Fabricating extracted data** - Reporting incorrect sample sizes, outcomes, or statistics from cited papers

Manual verification takes educators 3-5 hours per review.
Our Solution:
Automated fraud detection in under 5 minutes using AI and programmatic search validation.
