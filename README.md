# EMR Pipeline (GoHealth Assessment) | James Ezeilo

## Overview

An ETL pipeline for processing Electronic Medical Records (EMR) data into a structured SQLite database for analytics and reporting.

## Design Decisions

_NOTE: Most design decisions were based off assumptions made of what the hypothetical analytics team would prefer such as formatting for dates, currency, phone numbers along with invalid values for names, addresses, etc._

- Separate transformation scripts/modules for easier debugging, testing, and overall maintenance  
- Logged validation issues using "INFO" instead of "DEBUG" mode to avoid exposing patient data  
- Used a single load script (instead of one per dataset) to minimize effort for the analytics team by importing all data together, especially since the datasets are related through foreign keys  
- Dates formatted using the ISO8601 standard for consistency and compatibility  
- U.S. area codes included for phone numbers  
- Invalid or malformed data removed and set to NaN, assuming the analytics team prefers clean datasets only  

## Design Process

- Sketched out a rough idea for the overall pipeline  
- Built an Excel-to-CSV extraction function to save raw data to the data/raw folder  
- Developed transformation functions for each dataset (patients, visits, labs, ICDs) and wrote output to a staged folder  
- After staging, designed a database schema with a table for each dataset, including primary and foreign keys  
- Created shared helper functions (for date validation and CSV loading) to reduce redundancy
- Developed and built upon pytests for validation functions to handle edge cases 

## Desired Additions

If given more time, I would:
- Convert the run_pipeline script into an Airflow DAG  
- Expand test coverage, including helper function tests and edge cases  
- Implement SCD and SQL schema migrations  
- Automate test runs using GitHub Actions on push/PR  
- Add a dockerfile to install dependencies and run the pipeline and tests

**Data Flow**
   ```
   Excel File (source) → CSV Extraction → Data Validation → SQLite Database
   ```

[View Full Architecture Diagram](https://www.canva.com/design/DAGpGApEZUk/4qgSzW7LkrOFZo-9N9Q4lg/view?utm_content=DAGpGApEZUk&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h34aca163d1)

![ERD](docs/emr_erd.png)

## Dependencies

- pandas (2.0.0) and numpy (1.24.0) for data manipulation and transformation  
- loguru (0.7.0) for logging  
- openpyxl (3.1.0) for Excel file handling  
- sqlite3 (Python standard library) for databasing  

---

If any feedback could be provided, I’d really appreciate it! I honestly just want to improve and get better wherever possible. Thank you for taking the time to review my assessment and consider me for the position!