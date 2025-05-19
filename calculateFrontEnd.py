from flask import Flask, request, jsonify, abort
from gcLibraryFrontEnd import calculateGraph

app = Flask(__name__)

@app.route('/calculateAlternative', methods=['POST'])
def handle_request():
    try:
        data = request.get_json()
        #
        requiredGraph = data.get('graphName')
        graph = data.get('graph')
        #with open('currentOperation.txt', 'w') as rfile:
        #    rfile.write(f"{requiredGraph}")
         
        
        resultGraph = calculateGraph(requiredGraph, graph)

        return jsonify({'graph' : resultGraph})

    except Exception as e:
        print(f'Произошла ошибка в обработке calculateGraph: {str(e)}')
        response = jsonify({'message' : str(e)})
        response.status_code = 400
        return response
        #abort(400, str(e))
        #abort(400, "К сожалению, произошла ошибкa")
        #return jsonify({'success' : False, 'error' : str(e)})

if __name__ == '__main__':
    app.run(host='localhost', port=6083, debug = True)
