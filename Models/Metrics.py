from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, \
    brier_score_loss, auc, confusion_matrix

def performance_metrics(testing_y, y_pred_binary):
    F1Macro = f1_score(testing_y, y_pred_binary, average='macro')
    PrecisionMacro = precision_score(testing_y, y_pred_binary, average='macro')
    RecallMacro = recall_score(testing_y, y_pred_binary, average='macro')
    Accuracy = accuracy_score(testing_y, y_pred_binary)
    ClassificationReport = classification_report(testing_y, y_pred_binary)
    CM = confusion_matrix(testing_y, y_pred_binary)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)

    performance_row = {
        "F1-Macro" : F1Macro,
        "Precision-Macro" : PrecisionMacro,
        "Recall-Macro" : RecallMacro,
        "Accuracy" : Accuracy,
        "ClassificationReport" : ClassificationReport,
        "ConfusionMatrix": CM,
        "PPV":PPV,
        "NPV":NPV
        #"Precision-Recall AUC": PRAUC
        #"BrierScoreProba" : BrierScoreProba,
        #"BrierScoreBinary" : BrierScoreBinary
    }

    return performance_row
