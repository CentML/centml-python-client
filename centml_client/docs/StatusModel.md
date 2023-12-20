# StatusModel


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**status** | [**CompilationStatus**](CompilationStatus.md) |  | 

## Example

```python
from centml_client.models.status_model import StatusModel

# TODO update the JSON string below
json = "{}"
# create an instance of StatusModel from a JSON string
status_model_instance = StatusModel.from_json(json)
# print the JSON string representation of the object
print StatusModel.to_json()

# convert the object into a dict
status_model_dict = status_model_instance.to_dict()
# create an instance of StatusModel from a dict
status_model_form_dict = status_model.from_dict(status_model_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


