
import Preprocessing.process_felix_dataset
import Preprocessing.perturb_stroke_cloud
import Preprocessing.convert_blenderAddOn
import Preprocessing.perturb_blenderAddOn

def run():

    folder = 'human'
    
    # Preprocessing.convert_blenderAddOn.convert(folder)
    # d_generator = Preprocessing.process_felix_dataset.cad2sketch_dataset_loader()
    # cad2sketch_generator = Preprocessing.cad2sketch_data_generator.cad2sketch_dataset_generator()

    d_generator = Preprocessing.perturb_blenderAddOn.perturbation_dataset_loader(folder)


if __name__ == "__main__":
    run()
