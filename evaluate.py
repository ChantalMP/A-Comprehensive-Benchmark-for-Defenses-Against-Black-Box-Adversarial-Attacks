from classifiers.mymodel import MyModel
from attack.foolbox_boundary_attack import run_attack
from utils.attack_testdata_loader import get_attack_testset
from utils.modelWrapper import ModelWrapper
from foolbox.attacks import BoundaryAttack, BoundaryAttackPlusPlus
from utils.image_utils import plot_result
from utils.config_loader import load_config
from utils.graph_and_value_generation import generate_graph_data_and_median_mean
import os
from defenses.rce_kDensity.rce_kDensity import RCE_KDensity
from defenses.defensegan.defensegan_model import DefGanModel
import numpy as np
import sys

class Evaluation():

    """
    class for doing evaluation measures and saving them into files
    dataset: string for used dataset either "cifar10" or "mnist"
    classifier_path and classifier_name: path and name of classification model
    runTimeDefense: string describing runtime defense if any
    attack: BoundaryAttack or BoundaryAttackPlusPlus
    """
    def __init__(self, dataset, classifier_path, classifier_name, runTimeDefense, detector, attack, model_type):
        self.dataset = dataset
        self.model_path = classifier_path
        self.model_name = classifier_name
        self.defense = runTimeDefense
        self.detector = detector
        self.attack = attack
        if classifier_name == "rce" or classifier_name == "ce":
            classifier = RCE_KDensity(classifier_name, dataset=dataset)
        elif classifier_name == "defgan_model":
            classifier = DefGanModel()
        else:
            classifier, _, _, _, _ = MyModel(dataset).get_model(model_name=self.model_name, model_path=self.model_path,
                                                                model_type=model_type)


        self.model = ModelWrapper(model_type, classifier, runTimeDefense=runTimeDefense, detector=detector, dataset=dataset, refset_name=refset_name)
        self.testdata = get_attack_testset(dataset)

    def init_attack(self, max_queries, distance):
        self.max_queries = max_queries
        self.distance = distance

    def evaluate_attack(self):
        orig_labels = []
        adv_labels = []
        images = []
        adversarials = []
        inputs, labels = self.testdata
        i = 0
        sucessful = 0
        if not os.path.isdir("evaluation_results/current_csv"):
            os.mkdir("evaluation_results/current_csv")
        if not os.path.isdir("evaluation_results/data"):
            os.mkdir("evaluation_results/data")
        if not os.path.isdir("evaluation_results/current_csv/{}".format(reference_name)):
            os.mkdir("evaluation_results/current_csv/{}".format(reference_name))
        for image, label in zip(inputs, labels):
            adversarial, done_queries = run_attack(image, label, self.model, -1, self.max_queries, self.distance, self.dataset, self.attack, refset_name=refset_name, model_name=reference_name)
            self.model.queries = 0
            if adversarial is None:
                os.remove("{}_log.csv".format(reference_name))
                i+=1
                continue
            os.rename("{}_log.csv".format(reference_name), "evaluation_results/current_csv/{}/{}_log{}.csv".format(reference_name, reference_name, i))
            adv_label = adversarial.adversarial_class
            orig_labels.append(label)
            adv_labels.append(adv_label)
            images.append(image)
            adversarials.append(adversarial.image)
            sucessful+=1
            i+=1

        generate_graph_data_and_median_mean(sucessful, self.max_queries, reference_name)
        np.save("evaluation_results/data/orig_labels_{}.npy".format(reference_name), orig_labels)
        np.save("evaluation_results/data/adv_labels_{}.npy".format(reference_name), adv_labels)
        np.save("evaluation_results/data/orig_image_{}.npy".format(reference_name), images)
        np.save("evaluation_results/data/adv_images_{}.npy".format(reference_name), adversarials)

        plot_result(orig_labels[:3], adv_labels[:3], images[:3], adversarials[:3], "{}.png".format(reference_name))
        print("results saved to attack_results/data/{}".format(reference_name))


if __name__ == '__main__':
    config_name = sys.argv[1]
    print("running config: {}".format(config_name))
    attack_dict = {"ba": BoundaryAttack, "bapp":BoundaryAttackPlusPlus}
    config = load_config(config_name)
    dataset = config['dataset']
    run_time_defense = config['run_time_defense']
    detector = config['detector']
    max_queries = config['max_queries']
    distance_measure = config['distance']
    classifier_name = config['classifier_name']
    classifier_path = "classifiers/{}_classifier/models/{}".format(dataset, classifier_name)
    reference_name = config['reference_name']
    attack = attack_dict[config['attack']]
    model_type = config['model_type']
    refset_name = config['refset_name']

    evaluation = Evaluation(dataset, classifier_path, classifier_name, run_time_defense, detector, attack, model_type=model_type)
    evaluation.init_attack(max_queries, distance_measure)
    evaluation.evaluate_attack()
