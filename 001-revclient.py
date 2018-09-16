from watson_machine_learning_client import WatsonMachineLearningAPIClient
import json


#Enter your Watson Machine learning service instance credentials here
wml_credentials = {
    "apikey": "***",
    "instance_id": "***",
    "password": "***",
    "url": "https://eu-gb.ml.cloud.ibm.com",
    "username": "***"
}
client = WatsonMachineLearningAPIClient(wml_credentials)
print("************WML CLIENT CREATED************")

# To get model details 
print("\n************** MODELS IN MY INSTANCE OF WML REPOSITORY **************")
client.repository.list_models()


print("\n************** INSTANCE DETAILS **************")
instance_details = client.service_instance.get_details()
print(json.dumps(instance_details.get("metadata"),indent=4))


print("\n************** MODEL DETAILS **************")
model_details = client.repository.get_model_details("2a6c4203-17e4-48b8-b854-59328635304a")
print(json.dumps(model_details.get("metadata"),indent=4))
print("Model Name:"+json.dumps(model_details.get("entity").get("name"),indent=4))
print("Runtime enviornment:"+json.dumps(model_details.get("entity").get("runtime_environment"),indent=4))


print("\n************** DEPLOYMENT DETAILS **************")
client.deployments.list()
deployment_details = client.deployments.get_details("aafcb105-816a-4c3b-a701-19600df0358c")
print(json.dumps(deployment_details.get("metadata"),indent=4))
scoring_endpoint = deployment_details.get("entity").get("scoring_url")
print("Scoring Endpoint:")
print(scoring_endpoint)


print("\n************** TEST SCORING **************")
payload_scoring = {"fields": ['INCOME','AGE_IN_YEARS','HAS_CHILDREN','LENGTH_OF_RESIDENCE','MARITAL_STATUS','HOME_OWNER_RENTER','NUMBER_OF_CHILDREN','EDUCATION','HOME_MARKET_VALUE',
                              'CREDIT_RATING','HOME_OWNER','NO_MARITAL_STATUS','COLLEGE_DEGREE','GOOD_CREDIT','CUST_TENURE','NOOFPRODUCTS'],
                   "values": [[80000,28,1,1,"Single","R",1,"Completed High School","150000 - 174999","500-549",0,0,0,1,0.5,1]]}

predictions=client.deployments.score(scoring_endpoint,payload_scoring)
print(json.dumps(predictions,indent=4))
