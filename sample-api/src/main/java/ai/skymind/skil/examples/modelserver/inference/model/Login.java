package ai.skymind.skil.examples.modelserver.inference.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public class Login {
    public static class Request {
        @JsonProperty("userId")
        public String userId;

        @JsonProperty("password")
        public String password;

        public Request() { }

        public Request(String userId, String password) {
            this.userId = userId;
            this.password = password;
        }
    }

    public static class Response {
        @JsonProperty("token")
        public String token;

        public Response() { }

        public Response(String token) {
            this.token = token;
        }
    }
}
