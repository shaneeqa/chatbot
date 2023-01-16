from flask import Flask, request, Response 
from flask_cors import CORS


app = Flask("__name__") # entry point
CORS(app)

@app.route("/", methods=['GET'])
def checkIfAPIIsWorkingFine():
    return Response("This is response from the chatbot app")
if __name__ == "__main__":
    app.run(debug = True)
