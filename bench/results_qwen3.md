# End-to-end tool selection

caller: `ollama:qwen3:8b`  
embedder: `fastembed:BAAI/bge-small-en-v1.5`  
k: 5  
cases: 290

| tools | condition | accuracy | correct | wrong | no-call | retrieval miss | accuracy if retrieved | tool tokens | ms |
|---|---|---|---|---|---|---|---|---|---|
| 50 | full | 0.466 | 135 | 3 | 152 | 0 | 0.466 | 1,776 | 1754 |
| 50 | toolbroker k=5 | 0.521 | 151 | 5 | 134 | 37 | 0.597 | 272 | 2175 |
| 200 | full | 0.300 | 87 | 2 | 201 | 0 | 0.300 | 6,734 | 3111 |
| 200 | toolbroker k=5 | 0.500 | 145 | 5 | 140 | 50 | 0.604 | 271 | 2009 |

## Sample failures

A *retrieval miss* means no correct tool reached the model; anything
else means it had the right tool in front of it and chose otherwise.

- 50 tools, full: `what roles does this user have` → expected ['identity__list_permissions'], got `(no tool)` (caller abstained)
- 50 tools, full: `the customer wants their money back` → expected ['billing__issue_refund'], got `(no tool)` (caller abstained)
- 50 tools, full: `upgrade them to the annual plan` → expected ['billing__update_subscription'], got `(no tool)` (caller abstained)
- 50 tools, full: `do we have any of these left` → expected ['inventory__get_stock_level'], got `(no tool)` (caller abstained)
- 50 tools, toolbroker: `how many units are on hand for a SKU` → expected ['inventory__get_stock_level'], got `(no tool)` (caller abstained)
- 50 tools, toolbroker: `correct the recorded quantity for a SKU` → expected ['inventory__adjust_stock'], got `(no tool)` (caller abstained)
- 50 tools, toolbroker: `where is a shipment right now` → expected ['shipping__track_shipment'], got `(no tool)` (caller abstained)
- 50 tools, toolbroker: `edit fields on a contact` → expected ['crm__update_contact'], got `(no tool)` (caller abstained)
- 200 tools, full: `void an unpaid invoice` → expected ['billing__void_invoice'], got `(no tool)` (caller abstained)
- 200 tools, full: `change the billing plan for an account` → expected ['billing__update_subscription'], got `(no tool)` (caller abstained)
- 200 tools, full: `how many units are on hand for a SKU` → expected ['inventory__get_stock_level'], got `(no tool)` (caller abstained)
- 200 tools, full: `where is a shipment right now` → expected ['shipping__track_shipment'], got `(no tool)` (caller abstained)
- 200 tools, toolbroker: `how many units are on hand for a SKU` → expected ['inventory__get_stock_level'], got `(no tool)` (caller abstained)
- 200 tools, toolbroker: `correct the recorded quantity for a SKU` → expected ['inventory__adjust_stock'], got `(no tool)` (caller abstained)
- 200 tools, toolbroker: `where is a shipment right now` → expected ['shipping__track_shipment'], got `(no tool)` (caller abstained)
- 200 tools, toolbroker: `list compute instances in a region` → expected ['infrastructure__list_instances'], got `(no tool)` (caller abstained)
