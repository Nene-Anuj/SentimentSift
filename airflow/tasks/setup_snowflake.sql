USE ROLE ACCOUNTADMIN;

-- ----------------------------------------------------------------------------
-- Step #1: Create and assign role for FINAL Project
-- ----------------------------------------------------------------------------
SET MY_USER = CURRENT_USER();
CREATE OR REPLACE ROLE FINAL_ROLE;
GRANT ROLE FINAL_ROLE TO ROLE SYSADMIN;
GRANT ROLE FINAL_ROLE TO USER IDENTIFIER($MY_USER);
GRANT EXECUTE TASK ON ACCOUNT TO ROLE FINAL_ROLE;
GRANT MONITOR EXECUTION ON ACCOUNT TO ROLE FINAL_ROLE;
GRANT IMPORTED PRIVILEGES ON DATABASE SNOWFLAKE TO ROLE FINAL_ROLE;

-- ----------------------------------------------------------------------------
-- Step #2: Create Database and Warehouse for FINAL Project
-- ----------------------------------------------------------------------------
CREATE OR REPLACE DATABASE FINAL_DB;
GRANT OWNERSHIP ON DATABASE FINAL_DB TO ROLE FINAL_ROLE;

CREATE OR REPLACE WAREHOUSE FINAL_WH 
    WAREHOUSE_SIZE = XSMALL 
    AUTO_SUSPEND = 300 
    AUTO_RESUME = TRUE;
GRANT OWNERSHIP ON WAREHOUSE FINAL_WH TO ROLE FINAL_ROLE;

USE ROLE FINAL_ROLE;
USE WAREHOUSE FINAL_WH;
USE DATABASE FINAL_DB;
 
-- Schemas
CREATE OR REPLACE SCHEMA FINAL;

