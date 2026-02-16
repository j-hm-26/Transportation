# libraries
library(dplyr)
library(tidyverse)
library(readxl)
library(arrow)

# pull in data

operational_data <- read_delim("./BNSF_Focus/Operational_Data.csv", delim = ";")
# Filter to only contain data mentioning BNSF
  # We could later change these to compare BNSF to other rails
operational_data_BNSF <- operational_data %>%
  filter(if_any(where(is.character), ~ str_detect(., regex("BNSF", ignore_case = TRUE))))

# Filter to only contain data mentioning BNSF
injury_illness <- read_delim("./BNSF_Focus/Injury_Illness.csv", delim = ",")
injury_illness_BNSF <- injury_illness %>% filter(RAILROAD == "BNSF")


#data_dict <- read_xlsx("./BNSF_Focus/Form55_Data_Dictionary.xlsx")


##########################################################################################
# Cleaning weekly BNSF data from https://www.stb.gov/reports-data/rail-service-data/
##########################################################################################
raw_weekly <- read_xlsx("./BNSF_Focus/EP724 Consolidated Data through 2026-02-04.xlsx",
                        col_types = "text")
colnames(raw_weekly)[7:ncol(raw_weekly)] <- as.Date(
  as.numeric(colnames(raw_weekly)[7:ncol(raw_weekly)]),
  origin = "1899-12-30")

weekly_BNSF <- raw_weekly %>% rename(Railroad = `Railroad/\r\nRegion`) %>%
  filter(Railroad == "BNSF")

# Creating data frames for each measure

##########################################################################################
# Peeking at Waybill data
##########################################################################################
waybill_2023 <- read_parquet("./cleaned_waybill_data/waybill_data_2023.parquet")
