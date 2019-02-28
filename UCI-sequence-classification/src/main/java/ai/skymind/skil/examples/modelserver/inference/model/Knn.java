package ai.skymind.skil.examples.modelserver.inference.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Arrays;

/**
 * Created by Michael on 6/2/17.
 */
public class Knn {
    public static class Request {
        public Request(int k, String ndarray) {
            this.k = k;
            this.ndarray = ndarray;
        }

        @JsonProperty("ndarray")
        private String ndarray;

        @JsonProperty("k")
        private int k;

        public String getNdArray() {
            return this.ndarray;
        }

        public int getK() {
            return this.k;
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Response {
        public static class KnnResult {
            public KnnResult() {
            }

            @JsonProperty("index")
            private int index = 0;

            public int getIndex() {
                return this.index;
            }

            @JsonProperty("distance")
            private double distance = 0.0;

            public double getDistance() {
                return this.distance;
            }

            @JsonProperty("label")
            private String label = null;

            public String getLabel() {
                return this.label;
            }

            public String toString() {
                return String.format("{index=%d, distance=%.15f, label=%s}", this.index, this.distance, this.label);
            }
        }
 
        @JsonProperty("results")
        private List<KnnResult> results = null;

        public List<KnnResult> getResults() {
            return this.results;
        }
        public String toString() {
            return "results{" + Arrays.deepToString(this.results.toArray()) + "}";
        }
    }
}
