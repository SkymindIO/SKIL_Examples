package io.skymind.skil.tutorial;

import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.normalize.Normalize;

public class IrisData {
    public static final String COL1_SEPAL_LENGTH = "Sepal length";
    public static final String COL2_SEPAL_WIDTH = "Sepal width";
    public static final String COL3_PETAL_LENGTH = "Petal length";
    public static final String COL4_PETAL_WIDTH = "Petal width";
    public static final String COL5_LABEL = "Species";

    public static final int INPUT_COUNT = 4;
    public static final int TOTAL_COLUMN_COUNT = INPUT_COUNT + 1;
    public static final int LABEL_COUNT = 3;
    public static final int LABEL_INDEX = TOTAL_COLUMN_COUNT - 1;

    public static final Schema SCHEMA = new Schema.Builder()
            .addColumnsDouble(COL1_SEPAL_LENGTH, COL2_SEPAL_WIDTH, COL3_PETAL_LENGTH, COL4_PETAL_WIDTH)
            .addColumnInteger(COL5_LABEL)
            .build();

    public static TransformProcess trainTransform(DataAnalysis analysis) {
        return new TransformProcess.Builder(IrisData.SCHEMA)
                .normalize(IrisData.COL1_SEPAL_LENGTH, Normalize.Standardize, analysis)
                .normalize(IrisData.COL2_SEPAL_WIDTH, Normalize.Standardize, analysis)
                .normalize(IrisData.COL3_PETAL_LENGTH, Normalize.Standardize, analysis)
                .normalize(IrisData.COL4_PETAL_WIDTH, Normalize.Standardize, analysis)
                .build();
    }

    public static TransformProcess inferenceTransform(DataAnalysis analysis) {
        return new TransformProcess.Builder(IrisData.SCHEMA)
                .removeColumns(IrisData.COL5_LABEL)
                .normalize(IrisData.COL1_SEPAL_LENGTH, Normalize.Standardize, analysis)
                .normalize(IrisData.COL2_SEPAL_WIDTH, Normalize.Standardize, analysis)
                .normalize(IrisData.COL3_PETAL_LENGTH, Normalize.Standardize, analysis)
                .normalize(IrisData.COL4_PETAL_WIDTH, Normalize.Standardize, analysis)
                .build();
    }
}
