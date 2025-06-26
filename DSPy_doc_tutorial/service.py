import dspy
from tools import fetch_flight_info, fetch_itinerary, pick_flight, book_flight, cancel_itinerary, get_user_info, file_ticket

class DSPyAirlineCustomerSerice(dspy.Signature):
    """You are an airline customer service agent that helps user book and manage flights.

    You are given a list of tools to handle user request, and you should decide the right tool to use in order to
    fullfil users' request."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
                "Message that summarizes the process result, and the information users need, e.g., the "
                "confirmation_number if a new flight is booked."
            )
        )
    
agent = dspy.ReAct(
    DSPyAirlineCustomerSerice,
    tools = [
        fetch_flight_info,
        fetch_itinerary,
        pick_flight,
        book_flight,
        cancel_itinerary,
        get_user_info,
        file_ticket,           
    ]
)

