Notes:

python -m venv .venv
.venv\Scripts\activate.bat


How to run (starting from root folder):
1.  python .\logistics-customer-support\agent_v1.py

Alternate Approach 
1. from root folder> adk web

Possible issue:
- The Booking agent might get into loop

Test Messages:
*user: Hi
*user: My address is 73 Jacques Perstraat
*user: What can you do?
*user: Do you know my address?


Run:
1. Agent APIs
 Base Folder> adk api_server
 http://127.0.0.1:8000/docs

 Right App is  "logistics-customer-support"

2. Backend API
Base Folder\glide-logistics-app\back-end > uvicorn main:app --reload --port 9090

3. Client React App
