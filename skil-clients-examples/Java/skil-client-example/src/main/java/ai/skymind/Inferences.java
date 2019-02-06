package ai.skymind;

import ai.skymind.auth.ApiKeyAuth;
import ai.skymind.skil.DefaultApi;
import ai.skymind.skil.model.INDArray;
import ai.skymind.skil.model.LoginRequest;
import ai.skymind.skil.model.LoginResponse;
import ai.skymind.skil.model.Prediction;

import java.util.Arrays;

public class Inferences {
    public static void main(String[] args) {
        ApiClient apiClient = new ApiClient();
        apiClient.setBasePath("http://localhost:9008"); // Replace this with the host and port of your SKIL server, if required.

        DefaultApi apiInstance = new DefaultApi(apiClient);

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

            System.out.println(
                apiInstance.predict(
                    new Prediction()
                        .id("12345")
                        .needsPreProcessing(false)
                        .prediction(
                            new INDArray()
                                .shape(Arrays.asList(1, 784))
                                .data(Arrays.asList(new Float[784]))
                                .ordering(INDArray.OrderingEnum.C)
                        ),
                    "new_deployment", "default", "new_model"
                )
            );
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
