from Models.Utils import get_distribution_scalars
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import json
from matplotlib import colors as mcolors
import matplotlib.patches as mpatches
import re

class ClassificationReport:
    def __init__(self):
        configs = json.load(open('Configuration.json', 'r'))

        self.model_results = []
        self.num_models=0
        self.output_path =   configs['paths']['classification_report_path']
        self.label_dict = configs['data']['classification_label']

        self.colors = mcolors.XKCD_COLORS


    def add_model_result( self, label, y_true, y_pred_binary, best_threshold,
                          precision_rt, recall_rt,yhat ):
        new_result = ModelResult(label, y_true, y_pred_binary, best_threshold, precision_rt,
                                 recall_rt,yhat)
        self.model_results.append(new_result)
        self.num_models = self.num_models+1

    def plot_distributions_vs_aucs( self ):
        percents = []
        sizes = []
        pr_aucs = []
        roc_aucs = []
        outcomes = []

        model_subsets = [x for x in self.model_results if ("3D" not in x.label)]
        for rs in model_subsets:
            outcome = rs.label
            minority_percent = rs.get_minority_percentage()
            minority_percent_sizes = rs.get_minority_percentage()*800

            pr_auc = rs.get_pr_auc()
            roc_auc = rs.get_roc_auc()

            outcomes.append(outcome)
            percents.append(minority_percent)
            sizes.append(minority_percent_sizes)
            pr_aucs.append(pr_auc)
            roc_aucs.append(roc_auc)

        fig, ax = plt.subplots(figsize=(8, 8))

        y1 = pr_aucs
        y2 = roc_aucs
        y = y1+y2
        x1 = range(1, len(y1)+1)
        x2 = range(1, len(y2)+1)
        x = list(x1)+list(x2)
        colors1 = ['C0'] * len(y1)
        colors2 = ['C1'] * len(y2)
        colors = colors1+colors2

        sizes = sizes + sizes
        scatter = ax.scatter(x, y, c = colors, s=sizes, alpha=0.5, label="PR-AUC")

        #handles, labels = scatter.legend_elements(prop="colors", alpha=0.5)

        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.5)
        labels = [str(round((float(re.sub("[^0-9.]","",x))*(1/8)),2)) for x in labels]

        legend1 = ax.legend(handles, labels, loc="center right", title="Minority %")
        ax.add_artist(legend1)

        prauc_patch = mpatches.Patch(color='C0', label='ROC-AUC')
        prauc_patch.set_alpha(0.5)
        rocauc_patch = mpatches.Patch(color='C1', label='PR-AUC')
        rocauc_patch.set_alpha(0.5)
        legend2 = ax.legend(handles= [prauc_patch, rocauc_patch], loc="lower right", title="Performance")
        ax.add_artist(legend2)

        outcome_labels = [self.label_dict[k] for k in outcomes]
        seq_len = range(1, len(outcomes)+1)
        plt.title('Minority Distribution vs Performance')
        plt.xlabel('Outcome')
        plt.ylabel("Performance: ROC-AUC & PR-AUC")
        xticks = list(set(seq_len))
        plt.xticks(ticks = xticks, labels=outcome_labels)

        plt.savefig(self.output_path + "distribution_plot.pdf", bbox_inches='tight')


    def plot_pr_auc( self ):
        plt.figure(figsize=(8, 8))
        model_subsets = [x for x in self.model_results if ("3D" not in x.label)]

        for rs in model_subsets:
            pr_auc = auc(rs.recall_vector, rs.precision_vector)

            if 'Mortality' in rs.label:
                style = 'dashdot'
            else:
                style = 'dashed'

            plt.plot(rs.recall_vector, rs.precision_vector, linewidth=1.5,
                     linestyle=style, label=self.label_dict[rs.label]+' %0.3f' % pr_auc)
            plt.plot([0, 1], [1, 0], linewidth=1.5, linestyle='solid')

            plt.xlim([-0.01, 1])
            plt.ylim([0, 1.01])
            plt.legend(loc='lower left')
            plt.title(' Precision Recall Curve')
            plt.ylabel('Precision')
            plt.xlabel('Recall')

        plt.savefig(self.output_path + "pr_auc.pdf", bbox_inches='tight')

    def plot_auc( self ):
        plt.figure(figsize=(8, 8))
        model_subsets = [x for x in self.model_results if ("3D" not in x.label)]
        for rs in model_subsets :
            fpr, tpr, _ = roc_curve(rs.y_true, rs.y_pred)
            roc_auc = auc(fpr, tpr)
            if 'Mortality' in rs.label:
                style = 'dashdot'
            else:
                style = 'dashed'

            plt.plot(fpr, tpr, linewidth=1.5, linestyle = style, label=self.label_dict[rs.label]+' %0.3f' % roc_auc)
            plt.plot([0, 0], [1, 1], linestyle = 'solid', linewidth=1.5)

            plt.xlim([-0.01, 1])
            plt.ylim([0, 1.01])
            plt.legend(loc='lower right')
            plt.title(' ROC Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')

        plt.savefig(self.output_path + "auc.pdf", bbox_inches='tight')

    def compare_lstim_xgboost(self, lstm_praucs):
        xgb_praucs = []
        outcomes = []
        width = 0.35

        for ms in self.model_results:
            xgb_prauc = ms.get_pr_auc()
            xgb_praucs.append(xgb_prauc)
            outcomes.append(ms.label)

        outcome_labels = [self.label_dict[k] for k in outcomes]
        performance_differences = [x-y for (x,y) in zip(xgb_praucs,lstm_praucs)]
        xindices = range(1, len(outcomes)+1)

        plt.figure(figsize=(8, 8))
        p1 = plt.bar(xindices, lstm_praucs, width)
        p2 = plt.bar(xindices, performance_differences, width,
                     bottom=lstm_praucs)

        plt.ylabel('PR-AUC Contribution')
        plt.title('Performance Per and Modules and Outcomes')
        plt.xticks(xindices, outcome_labels)
        #plt.yticks(np.arange(0, 81, 10))
        plt.legend((p1[0], p2[0]), ('Dynamic-KD', 'Static-OP'))


        plt.savefig(self.output_path + "xgboost_vs_lstm.pdf", bbox_inches='tight')


class ModelResult:
    def __init__(self, label, y_true, y_pred_binary, best_threshold,
                 precision_rt, recall_rt,yhat ):
        self.label = label
        self.y_true = y_true
        self.y_pred = y_pred_binary
        self.threshold = best_threshold
        self.precision_vector = precision_rt
        self.recall_vector = recall_rt
        self.yhat = yhat

    def get_roc_auc (self):
        fpr, tpr, thresh = roc_curve(self.y_true, self.yhat)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def get_pr_auc (self):
        pr_auc =  auc(self.recall_vector, self.precision_vector)
        return pr_auc

    def get_minority_percentage( self ):
        distr = get_distribution_scalars(self.y_true)
        print(distr[1])
        return (distr[1])


