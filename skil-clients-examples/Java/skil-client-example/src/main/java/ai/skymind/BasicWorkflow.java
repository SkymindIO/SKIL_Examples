package ai.skymind;

import ai.skymind.auth.ApiKeyAuth;
import ai.skymind.skil.DefaultApi;
import ai.skymind.skil.model.*;

/**
 * Hello world!
 *
 */
public class BasicWorkflow 
{
    public static void main( String[] args ) {

        ApiClient apiClient = Configuration.getDefaultApiClient();
        apiClient.setBasePath("http://localhost:9008"); // Replace this with the host and port of your SKIL server if required

        DefaultApi apiInstance = new DefaultApi();

        try { // Login Successful
            LoginResponse loginResponse = apiInstance.login(
                    new LoginRequest()
                            .userId("admin")
                            .password("Password@123")
            );

            System.out.println(loginResponse);

            // Configure API key authorization: api_key
            ApiKeyAuth api_key = (ApiKeyAuth) apiClient.getAuthentication("api_key");
            api_key.setApiKeyPrefix("Bearer");
            api_key.setApiKey(loginResponse.getToken());


        } catch (ApiException e) {
            System.err.println("Exception when calling DefaultApi#login");
            e.printStackTrace();
        }
    }
}
