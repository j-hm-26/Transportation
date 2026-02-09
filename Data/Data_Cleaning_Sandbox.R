# libraries
library(dplyr)
library(tidyverse)
library(readxl)

# pull in data

federal_funding <- read_delim("./BNSF_Focus/Operational_Data.csv", delim = ";")
