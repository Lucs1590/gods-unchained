## Unreleased

### Feat

- load environment variables and configure Redis host from .env file
- remove 'lucas' from allowed_users list
- enhance Redis connection handling and add user to allowed_users
- add Docker Compose configuration for web and Redis services
- Remove test_set_strategy function

### Refactor

- restructure Docker Compose configuration to separate Redis service into infra.yaml

## 0.5.0 (2024-06-10)

### Feat

- add type and remove useless lib

### Fix

- src path
- path import
- spelling

### Refactor

- change summary list

## 0.4.0 (2024-06-09)

### Feat

- add missing sample step
- applying model building good practices
- using dataframe instead of df
- applying feature engineering good practices
- applying preprocessing good practices
- add model building step
- feature engineer step
- add initial preprocessing code
- add initial pipeline log

### Fix

- path and divide by zero errors
- divide by zero runtime error
- wrong space
- path reference
- script sorting

### Refactor

- changing logger name

## 0.3.0 (2024-06-08)

### Feat

- Add endpoint to retrieve card information by ID
- exporting raw data instead of calculated

### Refactor

- changing csv to parquet

## 0.2.0 (2024-06-08)

### Feat

- done eda
- add missing libs
- add discretizer view
- initial feature engineering and model building
- improving graphs, making analysis and normal test
- add data analysis and visualization
- dataset initial analysis
- initial structure to eda
- add application logging
- Add Pydantic model for strategy response
- applying card search
- Add prometheus metric
- add dummy to test the redis cache
- Mount Prometheus endpoint with WSGI middleware and redis server
- Improve error raise
- Add basic authentication to the api
- Change HTTP method for retrieving card strategy endpoint
- rename endpoint to retrieve_card_strategy
