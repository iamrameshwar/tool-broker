# End-to-end tool selection

caller: `lexical (offline)`  
embedder: `fastembed:BAAI/bge-small-en-v1.5`  
k: 5  
cases: 290

| tools | condition | accuracy | correct | wrong | no-call | retrieval miss | accuracy if retrieved | tool tokens | ms |
|---|---|---|---|---|---|---|---|---|---|
| 50 | full | 0.334 | 97 | 177 | 16 | 0 | 0.334 | 2,145 | 0 |
| 50 | toolbroker k=5 | 0.438 | 127 | 113 | 50 | 37 | 0.502 | 217 | 0 |
| 200 | full | 0.324 | 94 | 180 | 16 | 0 | 0.324 | 8,128 | 1 |
| 200 | toolbroker k=5 | 0.441 | 128 | 110 | 52 | 50 | 0.533 | 212 | 0 |
| 1000 | full | 0.338 | 98 | 176 | 16 | 0 | 0.338 | 40,618 | 4 |
| 1000 | toolbroker k=5 | 0.445 | 129 | 101 | 60 | 61 | 0.563 | 210 | 0 |

## Sample failures

A *retrieval miss* means no correct tool reached the model; anything
else means it had the right tool in front of it and chose otherwise.

- 50 tools, full: `where is a shipment right now` → expected ['shipping__track_shipment'], got `shipping__cancel_shipment` (caller chose wrong)
- 50 tools, full: `the customer wants their money back` → expected ['billing__issue_refund'], got `infrastructure__scale_deployment` (caller chose wrong)
- 50 tools, full: `bill this client for last month's work` → expected ['billing__create_invoice'], got `shipping__list_carriers` (caller chose wrong)
- 50 tools, full: `we sent that bill by mistake, undo it` → expected ['billing__void_invoice'], got `observability__get_trace` (caller chose wrong)
- 50 tools, toolbroker: `where is a shipment right now` → expected ['shipping__track_shipment'], got `shipping__cancel_shipment` (caller chose wrong)
- 50 tools, toolbroker: `the customer wants their money back` → expected ['billing__issue_refund'], got `inventory__adjust_stock` (caller chose wrong)
- 50 tools, toolbroker: `has this customer actually paid us` → expected ['billing__list_payments'], got `billing__void_invoice` (caller chose wrong)
- 50 tools, toolbroker: `do we have any of these left` → expected ['inventory__get_stock_level'], got `(no tool)` (caller abstained)
- 200 tools, full: `where is a shipment right now` → expected ['shipping__track_shipment'], got `shipping__cancel_shipment` (caller chose wrong)
- 200 tools, full: `the customer wants their money back` → expected ['billing__issue_refund'], got `infrastructure__scale_deployment` (caller chose wrong)
- 200 tools, full: `bill this client for last month's work` → expected ['billing__create_invoice'], got `shipping__list_carriers` (caller chose wrong)
- 200 tools, full: `we sent that bill by mistake, undo it` → expected ['billing__void_invoice'], got `observability__get_trace` (caller chose wrong)
- 200 tools, toolbroker: `where is a shipment right now` → expected ['shipping__track_shipment'], got `shipping__cancel_shipment` (caller chose wrong)
- 200 tools, toolbroker: `the customer wants their money back` → expected ['billing__issue_refund'], got `inventory__adjust_stock` (caller chose wrong)
- 200 tools, toolbroker: `has this customer actually paid us` → expected ['billing__list_payments'], got `billing__void_invoice` (caller chose wrong)
- 200 tools, toolbroker: `do we have any of these left` → expected ['inventory__get_stock_level'], got `(no tool)` (retrieval miss)
- 1000 tools, full: `where is a shipment right now` → expected ['shipping__track_shipment'], got `shipping__cancel_shipment` (caller chose wrong)
- 1000 tools, full: `the customer wants their money back` → expected ['billing__issue_refund'], got `infrastructure__scale_deployment` (caller chose wrong)
- 1000 tools, full: `bill this client for last month's work` → expected ['billing__create_invoice'], got `shipping__list_carriers` (caller chose wrong)
- 1000 tools, full: `we sent that bill by mistake, undo it` → expected ['billing__void_invoice'], got `observability__get_trace` (caller chose wrong)
- 1000 tools, toolbroker: `where is a shipment right now` → expected ['shipping__track_shipment'], got `shipping__cancel_shipment` (caller chose wrong)
- 1000 tools, toolbroker: `the customer wants their money back` → expected ['billing__issue_refund'], got `inventory__adjust_stock` (caller chose wrong)
- 1000 tools, toolbroker: `has this customer actually paid us` → expected ['billing__list_payments'], got `billing__void_invoice` (caller chose wrong)
- 1000 tools, toolbroker: `upgrade them to the annual plan` → expected ['billing__update_subscription'], got `(no tool)` (retrieval miss)
