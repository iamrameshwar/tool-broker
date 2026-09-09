"""Hand-authored labelled queries for the benchmark catalogue.

Written by hand rather than templated, because templated paraphrases measure
string similarity and nothing else. Each case is tagged with a *category*, so a
run reports which kinds of query a retriever fails on rather than one aggregate
that hides the answer.

Categories, roughly in order of difficulty:

``paraphrase``
    Restates what the tool does, in different words. The easy case.
``goal``
    What the user wants, in their own vocabulary. No tool words.
``jargon``
    Abbreviations and domain shorthand — SKU, SLA, MRR, p99.
``indirect``
    Describes a symptom or situation; the tool has to be inferred.
``distractor``
    Deliberately shares vocabulary with the *wrong* tool. This is where
    lexical retrieval collapses and where a real user's phrasing often lands.
``multi``
    Needs more than one tool. Any of the expected set counts as a hit, so
    this measures whether at least one relevant tool surfaced.
``ambiguous``
    More than one defensible answer; all acceptable ones are labelled.
``negative``
    No tool applies. The correct behaviour is to return nothing.

Labelling rule: a case lists every tool a competent engineer would accept, not
only the single "best" one. Scoring a defensible answer as a miss makes the
numbers look worse than the system is, which is just as misleading as the
reverse.
"""

from __future__ import annotations

# (query, expected tool ids, category)
Case = tuple[str, list[str], str]

QUERIES: list[Case] = [
    # ---------------- paraphrase -------------------------------------------
    ("refund a payment", ["billing/issue_refund"], "paraphrase"),
    ("raise a new invoice for an account", ["billing/create_invoice"], "paraphrase"),
    ("void an unpaid invoice", ["billing/void_invoice"], "paraphrase"),
    ("show payments received on an account", ["billing/list_payments"], "paraphrase"),
    ("change the billing plan for an account", ["billing/update_subscription"], "paraphrase"),
    ("how many units are on hand for a SKU", ["inventory/get_stock_level"], "paraphrase"),
    ("correct the recorded quantity for a SKU", ["inventory/adjust_stock"], "paraphrase"),
    ("which SKUs are below their reorder point", ["inventory/list_low_stock"], "paraphrase"),
    ("move units between warehouses", ["inventory/transfer_stock"], "paraphrase"),
    ("hold units against an order", ["inventory/reserve_stock"], "paraphrase"),
    ("book a shipment with a carrier", ["shipping/create_shipment"], "paraphrase"),
    ("where is a shipment right now", ["shipping/track_shipment"], "paraphrase"),
    ("cancel a shipment before it is picked up", ["shipping/cancel_shipment"], "paraphrase"),
    ("which carriers serve a destination", ["shipping/list_carriers"], "paraphrase"),
    ("predicted arrival date for a route", ["shipping/estimate_delivery"], "paraphrase"),
    ("add a user account", ["identity/create_user"], "paraphrase"),
    ("disable a user account", ["identity/deactivate_user"], "paraphrase"),
    ("send someone a password reset link", ["identity/reset_password"], "paraphrase"),
    ("issue a new API key and revoke the old one", ["identity/rotate_api_key"], "paraphrase"),
    ("what roles does this user have", ["identity/list_permissions"], "paraphrase"),
    ("search application logs over a time range", ["observability/query_logs"], "paraphrase"),
    ("time series for a service metric", ["observability/get_metrics"], "paraphrase"),
    ("which alerts are firing", ["observability/list_alerts"], "paraphrase"),
    ("mute an alert for a while", ["observability/silence_alert"], "paraphrase"),
    ("fetch a distributed trace by id", ["observability/get_trace"], "paraphrase"),
    ("restart a running service", ["infrastructure/restart_service"], "paraphrase"),
    ("change the replica count", ["infrastructure/scale_deployment"], "paraphrase"),
    ("list compute instances in a region", ["infrastructure/list_instances"], "paraphrase"),
    ("revert a service to its previous release", ["infrastructure/rollback_release"], "paraphrase"),
    ("roll out a build to an environment", ["infrastructure/deploy_release"], "paraphrase"),
    ("find contacts by email or company", ["crm/search_contacts"], "paraphrase"),
    ("open a sales opportunity", ["crm/create_deal"], "paraphrase"),
    ("record a call or meeting", ["crm/log_activity"], "paraphrase"),
    ("combine duplicate contact records", ["crm/merge_contacts"], "paraphrase"),
    ("edit fields on a contact", ["crm/update_contact"], "paraphrase"),
    ("find support tickets", ["support/search_tickets"], "paraphrase"),
    ("open a support ticket", ["support/create_ticket"], "paraphrase"),
    ("raise a ticket's priority and reassign it", ["support/escalate_ticket"], "paraphrase"),
    ("mark a ticket resolved", ["support/close_ticket"], "paraphrase"),
    ("tickets past their response deadline", ["support/list_sla_breaches"], "paraphrase"),
    # ---------------- goal --------------------------------------------------
    ("the customer wants their money back", ["billing/issue_refund"], "goal"),
    ("bill this client for last month's work", ["billing/create_invoice"], "goal"),
    ("we sent that bill by mistake, undo it", ["billing/void_invoice"], "goal"),
    ("has this customer actually paid us", ["billing/list_payments"], "goal"),
    ("upgrade them to the annual plan", ["billing/update_subscription"], "goal"),
    ("do we have any of these left", ["inventory/get_stock_level"], "goal"),
    ("the count in the system is wrong", ["inventory/adjust_stock"], "goal"),
    ("what do we need to reorder", ["inventory/list_low_stock"], "goal"),
    ("send stock from the London depot to Berlin", ["inventory/transfer_stock"], "goal"),
    ("set these units aside for this order", ["inventory/reserve_stock"], "goal"),
    ("get this parcel on its way", ["shipping/create_shipment"], "goal"),
    ("the customer wants to know where their parcel is", ["shipping/track_shipment"], "goal"),
    ("stop that delivery going out", ["shipping/cancel_shipment"], "goal"),
    ("who can deliver to rural Ireland", ["shipping/list_carriers"], "goal"),
    ("when will it get there", ["shipping/estimate_delivery"], "goal"),
    ("a new engineer is starting on Monday", ["identity/create_user"], "goal"),
    ("someone left the company today", ["identity/deactivate_user"], "goal"),
    ("they cannot get in and forgot their password", ["identity/reset_password"], "goal"),
    ("a secret got committed to a public repo", ["identity/rotate_api_key"], "goal"),
    ("can this person see the finance dashboard", ["identity/list_permissions"], "goal"),
    ("what happened around three this morning", ["observability/query_logs"], "goal"),
    ("show me how latency has trended this week", ["observability/get_metrics"], "goal"),
    ("what is currently broken", ["observability/list_alerts"], "goal"),
    ("stop paging me about this, we know", ["observability/silence_alert"], "goal"),
    ("follow this one request all the way through", ["observability/get_trace"], "goal"),
    ("turn it off and on again", ["infrastructure/restart_service"], "goal"),
    ("we need more capacity for the sale", ["infrastructure/scale_deployment"], "goal"),
    ("what machines do we have running", ["infrastructure/list_instances"], "goal"),
    ("that release broke everything, undo it", ["infrastructure/rollback_release"], "goal"),
    ("ship it to staging", ["infrastructure/deploy_release"], "goal"),
    ("who is our contact at that company", ["crm/search_contacts"], "goal"),
    ("this lead looks promising, track it", ["crm/create_deal"], "goal"),
    ("note that I spoke to them yesterday", ["crm/log_activity"], "goal"),
    ("we have the same person in here twice", ["crm/merge_contacts"], "goal"),
    ("their job title changed", ["crm/update_contact"], "goal"),
    ("has anyone else reported this problem", ["support/search_tickets"], "goal"),
    ("log this complaint properly", ["support/create_ticket"], "goal"),
    ("this needs a senior person right now", ["support/escalate_ticket"], "goal"),
    ("that is sorted, we can wrap it up", ["support/close_ticket"], "goal"),
    ("which customers have we kept waiting too long", ["support/list_sla_breaches"], "goal"),
    # ---------------- jargon ------------------------------------------------
    ("SKU on hand qty", ["inventory/get_stock_level"], "jargon"),
    ("stock take variance correction", ["inventory/adjust_stock"], "jargon"),
    ("ROP breach report", ["inventory/list_low_stock"], "jargon"),
    ("inter-DC stock movement", ["inventory/transfer_stock"], "jargon"),
    ("ATP allocation for an order", ["inventory/reserve_stock"], "jargon"),
    ("chargeback reversal", ["billing/issue_refund"], "jargon"),
    ("AR invoice raise", ["billing/create_invoice"], "jargon"),
    ("credit note against an unpaid invoice", ["billing/void_invoice"], "jargon"),
    ("MRR plan change", ["billing/update_subscription"], "jargon"),
    ("remittance history", ["billing/list_payments"], "jargon"),
    ("p99 latency series", ["observability/get_metrics"], "jargon"),
    ("grep prod stderr", ["observability/query_logs"], "jargon"),
    ("open pages right now", ["observability/list_alerts"], "jargon"),
    ("snooze the pager", ["observability/silence_alert"], "jargon"),
    ("span waterfall for a request id", ["observability/get_trace"], "jargon"),
    ("bounce the pods", ["infrastructure/restart_service"], "jargon"),
    ("bump replicas for the sale", ["infrastructure/scale_deployment"], "jargon"),
    ("roll back to the last known good", ["infrastructure/rollback_release"], "jargon"),
    ("cut a release to prod", ["infrastructure/deploy_release"], "jargon"),
    ("EC2 inventory by region", ["infrastructure/list_instances"], "jargon"),
    ("offboard the leaver", ["identity/deactivate_user"], "jargon"),
    ("onboard a joiner", ["identity/create_user"], "jargon"),
    ("creds rotation after a leak", ["identity/rotate_api_key"], "jargon"),
    ("RBAC grants for a principal", ["identity/list_permissions"], "jargon"),
    ("SLA breach queue", ["support/list_sla_breaches"], "jargon"),
    ("P1 escalation", ["support/escalate_ticket"], "jargon"),
    ("dedupe the CRM records", ["crm/merge_contacts"], "jargon"),
    ("log the touchpoint", ["crm/log_activity"], "jargon"),
    ("open a new opp", ["crm/create_deal"], "jargon"),
    ("ETA for the consignment", ["shipping/estimate_delivery"], "jargon"),
    # ---------------- indirect ----------------------------------------------
    (
        "a customer says they were charged twice",
        ["billing/issue_refund", "billing/list_payments"],
        "indirect",
    ),
    ("the finance team says this account was never billed", ["billing/create_invoice"], "indirect"),
    ("we quoted the wrong amount on that document", ["billing/void_invoice"], "indirect"),
    ("the warehouse count does not match the system", ["inventory/adjust_stock"], "indirect"),
    ("we keep running out of this item", ["inventory/list_low_stock"], "indirect"),
    (
        "a customer is asking why their order has not arrived",
        ["shipping/track_shipment"],
        "indirect",
    ),
    (
        "the customer changed their mind before it left the building",
        ["shipping/cancel_shipment"],
        "indirect",
    ),
    (
        "we cannot ship to that postcode with our usual courier",
        ["shipping/list_carriers"],
        "indirect",
    ),
    (
        "someone's laptop was stolen",
        ["identity/deactivate_user", "identity/rotate_api_key"],
        "indirect",
    ),
    ("a contractor's engagement ended last week", ["identity/deactivate_user"], "indirect"),
    ("an employee is locked out before a client call", ["identity/reset_password"], "indirect"),
    ("the site feels slow but nothing is down", ["observability/get_metrics"], "indirect"),
    (
        "customers report errors but our dashboard is green",
        ["observability/query_logs"],
        "indirect",
    ),
    (
        "my phone has been buzzing all night about a known issue",
        ["observability/silence_alert"],
        "indirect",
    ),
    (
        "one specific checkout hung and we do not know where",
        ["observability/get_trace"],
        "indirect",
    ),
    (
        "everything went bad right after this morning's deploy",
        ["infrastructure/rollback_release"],
        "indirect",
    ),
    (
        "the service is wedged and not recovering on its own",
        ["infrastructure/restart_service"],
        "indirect",
    ),
    ("black friday traffic starts in an hour", ["infrastructure/scale_deployment"], "indirect"),
    (
        "sales says they have been emailing two different people at the same company",
        ["crm/merge_contacts"],
        "indirect",
    ),
    ("a prospect finally replied and wants pricing", ["crm/create_deal"], "indirect"),
    ("the customer is furious and threatening to leave", ["support/escalate_ticket"], "indirect"),
    (
        "legal wants to know how long customers waited last quarter",
        ["support/list_sla_breaches"],
        "indirect",
    ),
    (
        "the bug the customer reported has been fixed and released",
        ["support/close_ticket"],
        "indirect",
    ),
    (
        "a caller says they emailed support last week and heard nothing",
        ["support/search_tickets"],
        "indirect",
    ),
    (
        "we are about to promise a delivery date on a call",
        ["shipping/estimate_delivery"],
        "indirect",
    ),
    ("a big order came in and we must not oversell", ["inventory/reserve_stock"], "indirect"),
    ("the Berlin shop is empty and London has hundreds", ["inventory/transfer_stock"], "indirect"),
    (
        "someone downgraded and is still being charged the old rate",
        ["billing/update_subscription"],
        "indirect",
    ),
    (
        "a new starter cannot access anything on day one",
        ["identity/create_user", "identity/list_permissions"],
        "indirect",
    ),
    ("we need proof of what this account has paid us", ["billing/list_payments"], "indirect"),
    # ---------------- distractor --------------------------------------------
    # Each of these shares vocabulary with a tool that is NOT the answer.
    ("cancel the invoice, not the shipment", ["billing/void_invoice"], "distractor"),
    ("cancel the shipment, the invoice is fine", ["shipping/cancel_shipment"], "distractor"),
    ("close the user account, not the ticket", ["identity/deactivate_user"], "distractor"),
    ("close the ticket, the account stays open", ["support/close_ticket"], "distractor"),
    ("list what is running, not what is alerting", ["infrastructure/list_instances"], "distractor"),
    ("list what is alerting, not what is running", ["observability/list_alerts"], "distractor"),
    ("undo the release, not the invoice", ["infrastructure/rollback_release"], "distractor"),
    ("undo the invoice, the release is fine", ["billing/void_invoice"], "distractor"),
    ("search the tickets, not the contacts", ["support/search_tickets"], "distractor"),
    ("search the contacts, not the tickets", ["crm/search_contacts"], "distractor"),
    ("move the stock, do not change the count", ["inventory/transfer_stock"], "distractor"),
    ("change the count, do not move anything", ["inventory/adjust_stock"], "distractor"),
    ("reset the password, do not rotate the key", ["identity/reset_password"], "distractor"),
    ("rotate the key, the password is fine", ["identity/rotate_api_key"], "distractor"),
    ("track the parcel, do not book a new one", ["shipping/track_shipment"], "distractor"),
    ("book a new parcel, the old one is delivered", ["shipping/create_shipment"], "distractor"),
    ("refund the payment, do not void the invoice", ["billing/issue_refund"], "distractor"),
    ("scale the deployment, do not restart it", ["infrastructure/scale_deployment"], "distractor"),
    ("restart it, do not scale it", ["infrastructure/restart_service"], "distractor"),
    ("silence the alert, do not fix the service", ["observability/silence_alert"], "distractor"),
    ("update the contact, do not merge anything", ["crm/update_contact"], "distractor"),
    ("merge the contacts, do not edit them", ["crm/merge_contacts"], "distractor"),
    ("escalate it, do not close it", ["support/escalate_ticket"], "distractor"),
    (
        "what stock is low, not what is out of stock right now",
        ["inventory/list_low_stock"],
        "distractor",
    ),
    ("logs for the deploy, not the deploy itself", ["observability/query_logs"], "distractor"),
    ("metrics about shipments, not shipment tracking", ["observability/get_metrics"], "distractor"),
    ("permissions for the user, do not change them", ["identity/list_permissions"], "distractor"),
    ("payments on the account, not a new invoice", ["billing/list_payments"], "distractor"),
    ("who the carriers are, not where the parcel is", ["shipping/list_carriers"], "distractor"),
    ("record the call, do not open a deal", ["crm/log_activity"], "distractor"),
    # ---------------- multi -------------------------------------------------
    (
        "find the customer's order and refund it",
        ["billing/list_payments", "billing/issue_refund"],
        "multi",
    ),
    (
        "someone left: disable them and rotate their keys",
        ["identity/deactivate_user", "identity/rotate_api_key"],
        "multi",
    ),
    (
        "the deploy broke prod, roll back and tell me what the logs said",
        ["infrastructure/rollback_release", "observability/query_logs"],
        "multi",
    ),
    (
        "check stock then reserve it for this order",
        ["inventory/get_stock_level", "inventory/reserve_stock"],
        "multi",
    ),
    (
        "book the shipment and tell the customer when it arrives",
        ["shipping/create_shipment", "shipping/estimate_delivery"],
        "multi",
    ),
    (
        "look up the ticket and escalate it",
        ["support/search_tickets", "support/escalate_ticket"],
        "multi",
    ),
    (
        "find the duplicate contacts and merge them",
        ["crm/search_contacts", "crm/merge_contacts"],
        "multi",
    ),
    (
        "silence the alert then look at the metrics",
        ["observability/silence_alert", "observability/get_metrics"],
        "multi",
    ),
    (
        "scale up and then confirm the instances came online",
        ["infrastructure/scale_deployment", "infrastructure/list_instances"],
        "multi",
    ),
    (
        "void the invoice and raise a corrected one",
        ["billing/void_invoice", "billing/create_invoice"],
        "multi",
    ),
    (
        "onboard the new hire and check what access they got",
        ["identity/create_user", "identity/list_permissions"],
        "multi",
    ),
    (
        "we are low on this SKU, move some from another warehouse",
        ["inventory/list_low_stock", "inventory/transfer_stock"],
        "multi",
    ),
    (
        "cancel the shipment and refund the customer",
        ["shipping/cancel_shipment", "billing/issue_refund"],
        "multi",
    ),
    (
        "close the ticket and log the call against the contact",
        ["support/close_ticket", "crm/log_activity"],
        "multi",
    ),
    (
        "which alerts are firing and what does the trace show",
        ["observability/list_alerts", "observability/get_trace"],
        "multi",
    ),
    (
        "open a ticket and escalate it straight away",
        ["support/create_ticket", "support/escalate_ticket"],
        "multi",
    ),
    (
        "check they paid before you ship anything",
        ["billing/list_payments", "shipping/create_shipment"],
        "multi",
    ),
    (
        "reset their password and confirm their roles",
        ["identity/reset_password", "identity/list_permissions"],
        "multi",
    ),
    (
        "restart it and watch the error rate",
        ["infrastructure/restart_service", "observability/get_metrics"],
        "multi",
    ),
    (
        "update the contact and open a deal for them",
        ["crm/update_contact", "crm/create_deal"],
        "multi",
    ),
    # ---------------- ambiguous ---------------------------------------------
    ("cancel it", ["billing/void_invoice", "shipping/cancel_shipment"], "ambiguous"),
    ("undo that", ["billing/void_invoice", "infrastructure/rollback_release"], "ambiguous"),
    ("close it", ["support/close_ticket", "identity/deactivate_user"], "ambiguous"),
    ("what is the status", ["shipping/track_shipment", "observability/list_alerts"], "ambiguous"),
    (
        "search for it",
        ["support/search_tickets", "crm/search_contacts", "observability/query_logs"],
        "ambiguous",
    ),
    (
        "make a new one",
        ["support/create_ticket", "billing/create_invoice", "identity/create_user"],
        "ambiguous",
    ),
    ("this is urgent", ["support/escalate_ticket"], "ambiguous"),
    (
        "something is wrong with the account",
        ["billing/list_payments", "identity/list_permissions"],
        "ambiguous",
    ),
    ("fix the numbers", ["inventory/adjust_stock", "billing/void_invoice"], "ambiguous"),
    ("the customer is unhappy", ["support/escalate_ticket", "billing/issue_refund"], "ambiguous"),
    ("who is this person", ["crm/search_contacts", "identity/list_permissions"], "ambiguous"),
    (
        "stop it",
        [
            "shipping/cancel_shipment",
            "observability/silence_alert",
            "infrastructure/restart_service",
        ],
        "ambiguous",
    ),
    (
        "show me the history",
        ["billing/list_payments", "crm/log_activity", "observability/query_logs"],
        "ambiguous",
    ),
    ("we need more", ["infrastructure/scale_deployment", "inventory/transfer_stock"], "ambiguous"),
    (
        "check it before we commit",
        ["inventory/get_stock_level", "shipping/estimate_delivery"],
        "ambiguous",
    ),
    # ---------------- negative ----------------------------------------------
    ("who won the 1974 world cup", [], "negative"),
    ("translate this poem into old norse", [], "negative"),
    ("what is the airspeed velocity of an unladen swallow", [], "negative"),
    ("write me a haiku about deployment", [], "negative"),
    ("what is the capital of Mongolia", [], "negative"),
    ("explain quantum entanglement to a child", [], "negative"),
    ("recommend a restaurant in Lisbon", [], "negative"),
    ("what year did the Berlin wall come down", [], "negative"),
    ("convert 40 celsius to fahrenheit", [], "negative"),
    ("summarise the plot of Moby Dick", [], "negative"),
    ("what is my horoscope today", [], "negative"),
    ("teach me to play the ukulele", [], "negative"),
    ("what is the tallest mountain in Africa", [], "negative"),
    ("draft a resignation letter", [], "negative"),
    ("who painted the Mona Lisa", [], "negative"),
    ("how do I make sourdough starter", [], "negative"),
    ("what is the speed of light in a vacuum", [], "negative"),
    ("name three novels by Ursula K Le Guin", [], "negative"),
    ("what time zone is Reykjavik in", [], "negative"),
    ("compose a limerick about YAML", [], "negative"),
    # ---------------- paraphrase (second pass) ------------------------------
    ("give money back for a charge", ["billing/issue_refund"], "paraphrase"),
    ("issue a bill to a customer", ["billing/create_invoice"], "paraphrase"),
    ("stock count for a product code", ["inventory/get_stock_level"], "paraphrase"),
    ("put a hold on inventory", ["inventory/reserve_stock"], "paraphrase"),
    ("arrange a courier collection", ["shipping/create_shipment"], "paraphrase"),
    ("delivery status lookup", ["shipping/track_shipment"], "paraphrase"),
    ("turn off someone's login", ["identity/deactivate_user"], "paraphrase"),
    ("read the service logs", ["observability/query_logs"], "paraphrase"),
    ("bring a service back up", ["infrastructure/restart_service"], "paraphrase"),
    ("look up a customer record", ["crm/search_contacts"], "paraphrase"),
    # ---------------- goal (second pass) ------------------------------------
    ("they were double charged and want it sorted", ["billing/issue_refund"], "goal"),
    ("accounts wants this on their statement", ["billing/create_invoice"], "goal"),
    ("can we promise same day on this", ["shipping/estimate_delivery"], "goal"),
    ("the courier never picked it up, kill it", ["shipping/cancel_shipment"], "goal"),
    ("give the intern access", ["identity/create_user"], "goal"),
    ("is the API getting slower", ["observability/get_metrics"], "goal"),
    ("we are getting hammered, add servers", ["infrastructure/scale_deployment"], "goal"),
    ("that customer just signed, log it", ["crm/create_deal"], "goal"),
    ("this complaint needs a manager", ["support/escalate_ticket"], "goal"),
    ("how many people did we let down last month", ["support/list_sla_breaches"], "goal"),
    # ---------------- jargon (second pass) ----------------------------------
    ("COGS adjustment on hand count", ["inventory/adjust_stock"], "jargon"),
    ("3PL booking", ["shipping/create_shipment"], "jargon"),
    ("POD tracking number", ["shipping/track_shipment"], "jargon"),
    ("IAM principal deactivation", ["identity/deactivate_user"], "jargon"),
    ("HPA replica bump", ["infrastructure/scale_deployment"], "jargon"),
    # ---------------- indirect (second pass) --------------------------------
    (
        "our biggest customer is threatening to churn over a billing error",
        ["billing/issue_refund", "billing/list_payments"],
        "indirect",
    ),
    (
        "the auditor wants evidence of every charge on this account",
        ["billing/list_payments"],
        "indirect",
    ),
    (
        "we promised next-day and the courier has not scanned it",
        ["shipping/track_shipment"],
        "indirect",
    ),
    (
        "an ex-employee's token is still calling our API",
        ["identity/rotate_api_key", "identity/deactivate_user"],
        "indirect",
    ),
    (
        "error rate spiked at exactly the time of the last change",
        ["infrastructure/rollback_release"],
        "indirect",
    ),
    (
        "one region is fine and the other is timing out",
        ["observability/get_metrics", "observability/query_logs"],
        "indirect",
    ),
    (
        "the same alert has fired forty times tonight for a known cause",
        ["observability/silence_alert"],
        "indirect",
    ),
    ("marketing keeps mailing the same human twice", ["crm/merge_contacts"], "indirect"),
    (
        "we oversold the item and now cannot fulfil",
        ["inventory/get_stock_level", "inventory/adjust_stock"],
        "indirect",
    ),
    (
        "a customer says nobody has replied to them in a week",
        ["support/search_tickets", "support/list_sla_breaches"],
        "indirect",
    ),
    # ---------------- distractor (second pass) ------------------------------
    (
        "deploy nothing, just show me what is deployed",
        ["infrastructure/list_instances"],
        "distractor",
    ),
    ("do not restart it, tell me why it is failing", ["observability/query_logs"], "distractor"),
    ("not a refund, just show me what they paid", ["billing/list_payments"], "distractor"),
    ("not a new ticket, find the existing one", ["support/search_tickets"], "distractor"),
    ("not a new user, just their permissions", ["identity/list_permissions"], "distractor"),
    ("do not cancel it, just tell me where it is", ["shipping/track_shipment"], "distractor"),
    ("do not move stock, just tell me what is low", ["inventory/list_low_stock"], "distractor"),
    ("not the invoice, the subscription", ["billing/update_subscription"], "distractor"),
    ("not the alert, the underlying trace", ["observability/get_trace"], "distractor"),
    ("not a deal, just note the conversation", ["crm/log_activity"], "distractor"),
    # ---------------- multi (second pass) -----------------------------------
    (
        "check the low stock list and reorder by transferring from the other site",
        ["inventory/list_low_stock", "inventory/transfer_stock"],
        "multi",
    ),
    ("refund them and close the ticket", ["billing/issue_refund", "support/close_ticket"], "multi"),
    (
        "track the parcel and if it is lost, refund",
        ["shipping/track_shipment", "billing/issue_refund"],
        "multi",
    ),
    (
        "rotate the key then confirm what that user can still reach",
        ["identity/rotate_api_key", "identity/list_permissions"],
        "multi",
    ),
    (
        "deploy it and watch the alerts",
        ["infrastructure/deploy_release", "observability/list_alerts"],
        "multi",
    ),
    (
        "find the contact and open a ticket for them",
        ["crm/search_contacts", "support/create_ticket"],
        "multi",
    ),
    (
        "void the old invoice and change their plan",
        ["billing/void_invoice", "billing/update_subscription"],
        "multi",
    ),
    (
        "get the trace then check the logs around it",
        ["observability/get_trace", "observability/query_logs"],
        "multi",
    ),
    (
        "reserve the stock and book the courier",
        ["inventory/reserve_stock", "shipping/create_shipment"],
        "multi",
    ),
    (
        "disable the account and log why on the contact",
        ["identity/deactivate_user", "crm/log_activity"],
        "multi",
    ),
    # ---------------- ambiguous (second pass) -------------------------------
    ("can you check that", ["observability/list_alerts", "shipping/track_shipment"], "ambiguous"),
    ("sort it out", ["support/escalate_ticket", "infrastructure/restart_service"], "ambiguous"),
    ("it is wrong", ["inventory/adjust_stock", "crm/update_contact"], "ambiguous"),
    ("give them access", ["identity/create_user", "identity/list_permissions"], "ambiguous"),
    ("how much", ["inventory/get_stock_level", "billing/list_payments"], "ambiguous"),
    # ---------------- negative (second pass) --------------------------------
    ("what is a good name for a golden retriever", [], "negative"),
    ("how many bones are in the human foot", [], "negative"),
    ("write a python function to reverse a string", [], "negative"),
    ("what is the exchange rate today", [], "negative"),
    ("tell me a joke about databases", [], "negative"),
]


def cases() -> list[dict[str, object]]:
    """Return every labelled case as a JSON-serialisable dict."""
    return [
        {"query": query, "expected": expected, "category": category}
        for query, expected, category in QUERIES
    ]


def category_counts() -> dict[str, int]:
    """How many cases each category contributes."""
    counts: dict[str, int] = {}
    for _, _, category in QUERIES:
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))
