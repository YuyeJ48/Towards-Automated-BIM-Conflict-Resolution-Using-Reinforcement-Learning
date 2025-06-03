import requests

solibri_server_url = 'http://localhost:10876/solibri/v1'
url = f'{solibri_server_url}/models'
response = requests.get(url, headers={'accept': 'application/json'})
response_body = response.json()
print(response_body)