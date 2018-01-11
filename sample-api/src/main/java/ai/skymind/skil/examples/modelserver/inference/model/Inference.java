package ai.skymind.skil.examples.modelserver.inference.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.*;

/**
 * Created by Michael on 6/2/17.
 */
public class Inference {
    public static class Request {
        public Request(String ndArray) {
            this.ndArray = ndArray;
        }

        @JsonProperty("id")
        private String uuid = UUID.randomUUID().toString();

        @JsonIgnore
        private String ndArray;

        @JsonProperty("prediction")
        public Map<String, String> getPrediction() {
            final Map<String, String> prediction = new HashMap<String, String>();
            prediction.put("array", this.ndArray);

            return prediction;
        }
    }

    public interface Response {

        class Classify implements Response {
            @JsonProperty("results")
            private int[] results = null;

            @JsonProperty("probabilities")
            private float[] probabilities = null;

            public String toString() {
                return "Inference.Response.Classify{" +
                        "results{" + Arrays.toString(this.results) + "}, " +
                        "probabilities{" + Arrays.toString(this.probabilities) + "}";
            }
        }

        class MultiClassify implements Response {
            @JsonProperty("rankedOutcomes")
            private List<String[]> rankedOutcomes = null;

            @JsonProperty("maxOutcomes")
            private String[] maxOutcomes = null;

            @JsonProperty("probabilities")
            private List<float[]> probabilities = null;

            public String toString() {
                final StringBuilder sb = new StringBuilder();

                sb.append("Inference.Response.MultiClassify{");
                sb.append("rankedOutcomes{").append(listToString(this.rankedOutcomes)).append("}, ");
                sb.append("maxOutcomes{").append(Arrays.toString(this.maxOutcomes)).append("}, ");
                sb.append("probabilities{").append(listToString(this.probabilities)).append("}");
                sb.append("}");

                return sb.toString();
            }

            private String listToString(List l) {
                if (l == null) {
                    return "null";
                }

                return Arrays.deepToString(l.toArray());
            }
        }

    }
}
