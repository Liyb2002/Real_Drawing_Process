
import Preprocessing.process_felix_dataset
import Preprocessing.perturb_stroke_cloud


def run():
    
    d_generator = Preprocessing.process_felix_dataset.cad2sketch_dataset_loader()
    # cad2sketch_generator = Preprocessing.cad2sketch_data_generator.cad2sketch_dataset_generator()

    # d_generator = Preprocessing.perturb_stroke_cloud.perturbation_dataset_loader()


if __name__ == "__main__":
    run()
