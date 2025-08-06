JSON_SCHEMA_COMPANY_INCOME_STATEMENT="""{
    "$defs": {
        "incomeStatement": {
            "properties": {
                "company_name": {
                    "title": "Company Name",
                    "type": "string"
                },
                "year": {
                    "title": "Year",
                    "type": "string"
                },
                "revenue": {
                    "title": "Revenue",
                    "type": "number"
                },
                "gross_operating_profit_loss": {
                    "title": "Gross Operating Profit Loss",
                    "type": "number"
                },
                "operating_profit_loss": {
                    "title": "Operating Profit Loss",
                    "type": "number"
                },
                "net_financial_expense": {
                    "title": "Net Financial Expense",
                    "type": "number"
                },
                "net_non_recurring_income_expense": {
                    "title": "Net Non Recurring Income Expense",
                    "type": "number"
                },
                "pre_tax_profit_loss": {
                    "title": "Pre Tax Profit Loss",
                    "type": "number"
                },
                "profit_loss_year": {
                    "title": "Profit Loss Year",
                    "type": "number"
                }
            },
            "required": [
                "company_name",
                "year",
                "revenue",
                "gross_operating_profit_loss",
                "operating_profit_loss",
                "net_financial_expense",
                "net_non_recurring_income_expense",
                "pre_tax_profit_loss",
                "profit_loss_year"
            ],
            "title": "incomeStatement",
            "type": "object"
        }
    },
    "properties": {
        "incomeStatements": {
            "items": {
                "$ref": "#/$defs/incomeStatement"
            },
            "title": "Incomestatements",
            "type": "array",
            "maxItems": 8,
            "minItems": 8
        }
    },
    "required": [
        "incomeStatements"
    ],
    "title": "extractedData",
    "type": "object"
}"""

#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################

JSON_SCHEMA_EXTRACTEDDATA="""{
  "$defs": {
    "incomeStatement": {
      "properties": {
        "company_name": {
          "title": "Company Name",
          "type": "string"
        },
        "year": {
          "title": "Year",
          "type": "string"
        },
        "revenue": {
          "title": "Revenue",
          "type": "number"
        },
        "gross_operating_profit_loss": {
          "title": "Gross Operating Profit Loss",
          "type": "number"
        },
        "operating_profit_loss": {
          "title": "Operating Profit Loss",
          "type": "number"
        },
        "net_financial_expense": {
          "title": "Net Financial Expense",
          "type": "number"
        },
        "net_non_recurring_income_expense": {
          "title": "Net Non Recurring Income Expense",
          "type": "number"
        },
        "pre_tax_profit_loss": {
          "title": "Pre Tax Profit Loss",
          "type": "number"
        },
        "profit_loss_year": {
          "title": "Profit Loss Year",
          "type": "number"
        }
      },
      "required": [
        "company_name",
        "year",
        "revenue",
        "gross_operating_profit_loss",
        "operating_profit_loss",
        "net_financial_expense",
        "net_non_recurring_income_expense",
        "pre_tax_profit_loss",
        "profit_loss_year"
      ],
      "title": "incomeStatement",
      "type": "object"
    },
    "statementFinancialPosition": {
      "properties": {
        "Date_statement": {
          "title": "Date Statement",
          "type": "string"
        },
        "A": {
          "title": "A",
          "type": "number"
        },
        "A1": {
          "title": "A1",
          "type": "number"
        },
        "A2": {
          "title": "A2",
          "type": "number"
        },
        "A3": {
          "title": "A3",
          "type": "number"
        },
        "A4": {
          "title": "A4",
          "type": "number"
        },
        "A5": {
          "title": "A5",
          "type": "number"
        },
        "A6": {
          "title": "A6",
          "type": "number"
        },
        "A7": {
          "title": "A7",
          "type": "number"
        },
        "A8": {
          "title": "A8",
          "type": "number"
        },
        "A9": {
          "title": "A9",
          "type": "number"
        },
        "A10": {
          "title": "A10",
          "type": "number"
        },
        "A11": {
          "title": "A11",
          "type": "number"
        },
        "B": {
          "title": "B",
          "type": "number"
        },
        "B1": {
          "title": "B1",
          "type": "number"
        },
        "B2": {
          "title": "B2",
          "type": "number"
        },
        "B3": {
          "title": "B3",
          "type": "number"
        },
        "B4": {
          "title": "B4",
          "type": "number"
        },
        "C": {
          "title": "C",
          "type": "number"
        },
        "C1": {
          "title": "C1",
          "type": "number"
        },
        "C2": {
          "title": "C2",
          "type": "number"
        },
        "D": {
          "title": "D",
          "type": "number"
        },
        "D1": {
          "title": "D1",
          "type": "number"
        },
        "D2": {
          "title": "D2",
          "type": "number"
        },
        "E": {
          "title": "E",
          "type": "number"
        },
        "F": {
          "title": "F",
          "type": "number"
        },
        "F1": {
          "title": "F1",
          "type": "number"
        },
        "F2": {
          "title": "F2",
          "type": "number"
        },
        "F3": {
          "title": "F3",
          "type": "number"
        },
        "G": {
          "title": "G",
          "type": "number"
        },
        "G1": {
          "title": "G1",
          "type": "number"
        },
        "G2": {
          "title": "G2",
          "type": "number"
        },
        "G3": {
          "title": "G3",
          "type": "number"
        },
        "G4": {
          "title": "G4",
          "type": "number"
        },
        "TOTAL_ASSET": {
          "title": "Total Asset",
          "type": "number"
        },
        "H": {
          "title": "H",
          "type": "number"
        },
        "H1": {
          "title": "H1",
          "type": "number"
        },
        "H2": {
          "title": "H2",
          "type": "number"
        },
        "H3": {
          "title": "H3",
          "type": "number"
        },
        "I": {
          "title": "I",
          "type": "number"
        },
        "I1": {
          "title": "I1",
          "type": "number"
        },
        "I2": {
          "title": "I2",
          "type": "number"
        },
        "L": {
          "title": "L",
          "type": "number"
        },
        "L1": {
          "title": "L1",
          "type": "number"
        },
        "L2": {
          "title": "L2",
          "type": "number"
        },
        "M": {
          "title": "M",
          "type": "number"
        },
        "M1": {
          "title": "M1",
          "type": "number"
        },
        "M2": {
          "title": "M2",
          "type": "number"
        },
        "M3": {
          "title": "M3",
          "type": "number"
        },
        "M4": {
          "title": "M4",
          "type": "number"
        },
        "M5": {
          "title": "M5",
          "type": "number"
        },
        "TOTAL_LIABILITIES": {
          "title": "Total Liabilities",
          "type": "number"
        },
        "TOTAL_NET_ASSETS_VALUE_OF_THE_FUND": {
          "title": "Total Net Assets Value Of The Fund",
          "type": "number"
        },
        "NUMBER_OF_SUBSCRIPTED_UNITS": {
          "title": "Number Of Subscripted Units",
          "type": "number"
        },
        "UNIT_VALUE": {
          "title": "Unit Value",
          "type": "number"
        },
        "TOTAL_AMOUNT_DRAWN": {
          "title": "Total Amount Drawn",
          "type": "number"
        },
        "TOTAL_AMOUNT_TO_BE_DRAWN": {
          "title": "Total Amount To Be Drawn",
          "type": "number"
        },
        "TOTAL_AMOUNT_OF_REDEMPTIONS_MADE_CAPITAL": {
          "title": "Total Amount Of Redemptions Made Capital",
          "type": "number"
        },
        "TOTAL_AMOUNT_OF_DISTRUBUTIONS_MADE_INCOME": {
          "title": "Total Amount Of Distrubutions Made Income",
          "type": "number"
        },
        "TOTAL_AMOUNT_OF_SUBSCRIPTIONS_RECEIVED": {
          "title": "Total Amount Of Subscriptions Received",
          "type": "number"
        },
        "UNIT_VALUE_OF_SUBSCIBED_UNITS": {
          "title": "Unit Value Of Subscibed Units",
          "type": "number"
        }
      },
      "required": [
        "Date_statement",
        "A",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
        "A9",
        "A10",
        "A11",
        "B",
        "B1",
        "B2",
        "B3",
        "B4",
        "C",
        "C1",
        "C2",
        "D",
        "D1",
        "D2",
        "E",
        "F",
        "F1",
        "F2",
        "F3",
        "G",
        "G1",
        "G2",
        "G3",
        "G4",
        "TOTAL_ASSET",
        "H",
        "H1",
        "H2",
        "H3",
        "I",
        "I1",
        "I2",
        "L",
        "L1",
        "L2",
        "M",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "TOTAL_LIABILITIES",
        "TOTAL_NET_ASSETS_VALUE_OF_THE_FUND",
        "NUMBER_OF_SUBSCRIPTED_UNITS",
        "UNIT_VALUE",
        "TOTAL_AMOUNT_DRAWN",
        "TOTAL_AMOUNT_TO_BE_DRAWN",
        "TOTAL_AMOUNT_OF_REDEMPTIONS_MADE_CAPITAL",
        "TOTAL_AMOUNT_OF_DISTRUBUTIONS_MADE_INCOME",
        "TOTAL_AMOUNT_OF_SUBSCRIPTIONS_RECEIVED",
        "UNIT_VALUE_OF_SUBSCIBED_UNITS"
      ],
      "title": "statementFinancialPosition",
      "type": "object"
    }
  },
  "properties": {
    "statementFinancialPositions": {
      "items": {
        "$ref": "#/$defs/statementFinancialPosition"
      },
      "title": "Statementfinancialpositions",
      "type": "array",
      "maxItems": 2,
      "minItems":2
    },
    "incomeStatements": {
      "items": {
        "$ref": "#/$defs/incomeStatement"
      },
      "title": "Incomestatements",
      "type": "array",
      "maxItems": 8,
      "minItems":8      
    },
    "boardDirectorsChairman": {
      "title": "Boarddirectorschairman",
      "type": "string",
      "description": "the person who leads the internal oversight body responsible for supervising a company's financial and legal compliance."      
    },
    "boardStatutoryAuditorsChairman": {
      "title": "Boardstatutoryauditorschairman",
      "type": "string",
      "description": "the individual who presides over the board of directors, leading meetings and overseeing the board’s governance and strategic decisions"
    },
    "numberOfCompanies": {
     "title": "numberOfCompanies",
      "type": "number",
      "description": "the number of the companies held by the fund."  

    }
  },
  "required": [
    "statementFinancialPositions",
    "incomeStatements",
    "boardDirectorsChairman",
    "boardStatutoryAuditorsChairman",
    "numberOfCompanies"
  ],
  "title": "extractedData",
  "type": "object"
}"""


#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
JSON_SCHEMA_EASY=""" {
  "$defs": {
    "statementFinancialPosition": {
      "type": "object",
      "title": "statementFinancialPosition",
      "description": "",
      "properties": {
        "z": {
          "title": "Z",
          "description": "z refers to controlling investments",
          "type": "number"
        },
        "i": {
          "title": "I",
          "description": "i to non-controlling investments",
          "type": "number"
        },
        "k": {
          "title": "K",
          "description": "k to other equity instruments",
          "type": "number"
        },
        "g": {
          "title": "G",
          "description": "g to debt instruments",
          "type": "number"
        },
        "t": {
          "title": "T",
          "description": "t to OEIC units",
          "type": "number"
        }
      },
      "required": ["z", "i", "k", "g", "t"]
    }
  },
  "type": "object",
  "title": "Lista_Matteo",
  "description": "A list of general structures, each detailing the composition of financial instruments.",
  "properties": {
    "Lista_Matteo": {
      "type": "array",
      "title": "statementFinancialPositions",
      "minItems": 2,
      "maxItems": 2,
      "items": {
        "$ref": "#/$defs/statementFinancialPosition"
      }
    }
  },
  "required": ["Lista_Matteo"]
}"""

#############################################################################################################################################################àà
#############################################################################################################################################################àà
#############################################################################################################################################################àà
#############################################################################################################################################################àà
#############################################################################################################################################################àà
#############################################################################################################################################################àà

JSON_SCHEMA_D = """{
  "$defs": {
    "statementFinancialPosition": {
      "type": "object",
      "title": "statementFinancialPosition",
      "description": "",
      "properties": {
        "z": {
          "title": "Z",
          "description": "z refers to controlling investments",
          "type": "number"
        },
        "i": {
          "title": "I",
          "description": "i to non-controlling investments",
          "type": "number"
        },
        "k": {
          "title": "K",
          "description": "k to other equity instruments",
          "type": "number"
        },
        "g": {
          "title": "G",
          "description": "g to debt instruments",
          "type": "number"
        },
        "t": {
          "title": "T",
          "description": "t to OEIC units",
          "type": "number"
        }
      },
      "required": ["z", "i", "k", "g", "t"]
    }
  },
  "type": "object",
  "title": "Lista_Matteo",
  "description": "A list of general structures, each detailing the composition of financial instruments.",
  "properties": {
    "Lista_Matteo": {
      "type": "array",
      "title": "statementFinancialPositions",
      "minItems": 2,
      "maxItems": 8,
      "items": {
        "$ref": "#/$defs/statementFinancialPosition"
      }
    },
    "boardDirectorsChairman": {
      "type": "string",
      "description": "the person who leads the internal oversight body responsible for supervising a company's financial and legal compliance."
    },
    "boardStatutoryAuditorsChairman": {
      "type": "string",
      "description": "the individual who presides over the board of directors, leading meetings and overseeing the board’s governance and strategic decisions"
    }
  },
  "required": ["Lista_Matteo", "boardStatutoryAuditorsChairman", "boardDirectorsChairman"]
}"""

