# centml_client.DefaultApi

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**compile_model_handler_submit_model_id_post**](DefaultApi.md#compile_model_handler_submit_model_id_post) | **POST** /submit/{model_id} | Compile Model Handler
[**download_handler_download_model_id_get**](DefaultApi.md#download_handler_download_model_id_get) | **GET** /download/{model_id} | Download Handler
[**status_handler_status_model_id_get**](DefaultApi.md#status_handler_status_model_id_get) | **GET** /status/{model_id} | Status Handler


# **compile_model_handler_submit_model_id_post**
> compile_model_handler_submit_model_id_post(model_id, model, inputs)

Compile Model Handler

### Example

```python
import time
import os
import centml_client
from centml_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = centml_client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with centml_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = centml_client.DefaultApi(api_client)
    model_id = 'model_id_example' # str | 
    model = None # bytearray | 
    inputs = None # bytearray | 

    try:
        # Compile Model Handler
        api_instance.compile_model_handler_submit_model_id_post(model_id, model, inputs)
    except Exception as e:
        print("Exception when calling DefaultApi->compile_model_handler_submit_model_id_post: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **model_id** | **str**|  | 
 **model** | **bytearray**|  | 
 **inputs** | **bytearray**|  | 

### Return type

void (empty response body)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: multipart/form-data
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **download_handler_download_model_id_get**
> download_handler_download_model_id_get(model_id)

Download Handler

### Example

```python
import time
import os
import centml_client
from centml_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = centml_client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with centml_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = centml_client.DefaultApi(api_client)
    model_id = 'model_id_example' # str | 

    try:
        # Download Handler
        api_instance.download_handler_download_model_id_get(model_id)
    except Exception as e:
        print("Exception when calling DefaultApi->download_handler_download_model_id_get: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **model_id** | **str**|  | 

### Return type

void (empty response body)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **status_handler_status_model_id_get**
> StatusModel status_handler_status_model_id_get(model_id)

Status Handler

### Example

```python
import time
import os
import centml_client
from centml_client.models.status_model import StatusModel
from centml_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = centml_client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with centml_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = centml_client.DefaultApi(api_client)
    model_id = 'model_id_example' # str | 

    try:
        # Status Handler
        api_response = api_instance.status_handler_status_model_id_get(model_id)
        print("The response of DefaultApi->status_handler_status_model_id_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->status_handler_status_model_id_get: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **model_id** | **str**|  | 

### Return type

[**StatusModel**](StatusModel.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

