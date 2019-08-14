from imgaug import augmenters as iaa
import cv2
import os
import yaml
import itertools
import glob
import random

SEED = 6
random.seed(SEED)


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def parse_aug(aug_name, aug_obj):
    iaaOp = getattr(iaa, aug_name)
    augs = []
    image_name_suffixes = []
    if 'first' in aug_obj:
        param_values_dict = {}
        image_values_dict = {}
        aug_name_of_first = list(aug_obj['first'].keys())[0]
        augs_of_first, image_name_suffixes_of_first = parse_aug(aug_name_of_first, aug_obj['first'][aug_name_of_first])
        param_values_dict['first'] = augs_of_first
        image_values_dict['first'] = image_name_suffixes_of_first
        if 'second' in aug_obj:
            aug_name_of_second = list(aug_obj['second'].keys())[0]
            augs_of_second, image_name_suffixes_of_second = parse_aug(aug_name_of_second,
                                                                      aug_obj['second'][aug_name_of_second])
            param_values_dict['second'] = augs_of_second
            image_values_dict['second'] = image_name_suffixes_of_second
        param_value_list = list(product_dict(**param_values_dict))
        image_values_list = list(product_dict(**image_values_dict))
        for i in range(len(param_value_list)):
            augs.append(iaaOp(**param_value_list[i]))
            image_name_suffixes.append(
                aug_name + '_' + '_'.join(['{}_{}'.format(k, v) for k, v in image_values_list[i].items()]))
    elif '_LOOP' in aug_obj:
        for v in range(aug_obj['_LOOP']):
            augs.append(iaaOp())
            image_name_suffixes.append(aug_name + '_' + str(v))
    else:
        params = list(aug_obj.keys())
        param_values_dict = {}
        for param in params:
            if 'range' in aug_obj[param]:
                param_values_dict[param] = aug_obj[param]['range']
            else:
                ratio = 1 / aug_obj[param]['step'] if aug_obj[param]['step'] < 1 else 1
                param_values_dict[param] = [
                    value if 'requireInt' in aug_obj[param] else value / ratio for value in
                    range(int(float(aug_obj[param]['min']) * ratio),
                          int(float(aug_obj[param]['max']) * ratio),
                          int(float(aug_obj[param]['step']) * ratio))]
        param_value_list = list(product_dict(**param_values_dict))
        for pv in param_value_list:
            augs.append(iaaOp(**pv))
            image_name_suffixes.append(aug_name + '_' + '_'.join(['{}_{}'.format(k, v) for k, v in pv.items()]))
    return augs, image_name_suffixes


def random_aug(aug_name, aug_obj):
    iaaOp = getattr(iaa, aug_name)
    if 'first' in aug_obj:
        param_values_dict = {}
        image_values_dict = {}
        aug_name_of_first = list(aug_obj['first'].keys())[0]
        aug_of_first, image_name_suffix_of_first = random_aug(aug_name_of_first, aug_obj['first'][aug_name_of_first])
        param_values_dict['first'] = aug_of_first
        image_values_dict['first'] = image_name_suffix_of_first
        if 'second' in aug_obj:
            aug_name_of_second = list(aug_obj['second'].keys())[0]
            aug_of_second, image_name_suffix_of_second = random_aug(aug_name_of_second,
                                                                    aug_obj['second'][aug_name_of_second])
            param_values_dict['second'] = aug_of_second
            image_values_dict['second'] = image_name_suffix_of_second
        aug = iaaOp(**param_values_dict)
        image_name_suffix = aug_name + '_' + '_'.join(['{}_{}'.format(k, v) for k, v in image_values_dict.items()])
    elif '_LOOP' in aug_obj:
        aug = iaaOp()
        image_name_suffix = aug_name + '_' + str(random.randint(0, aug_obj['_LOOP']))
    else:
        params = list(aug_obj.keys())
        param_values_dict = {}
        for param in params:
            if 'range' in aug_obj[param]:
                param_values_dict[param] = random.choice(aug_obj[param]['range'])
            else:
                ratio = 1 / aug_obj[param]['step'] if aug_obj[param]['step'] < 1 else 1
                random_result = random.randrange(int(float(aug_obj[param]['min']) * ratio),
                                                 int(float(aug_obj[param]['max']) * ratio),
                                                 int(float(aug_obj[param]['step']) * ratio))
                param_values_dict[param] = random_result if 'requireInt' in aug_obj[param] else random_result / ratio
        aug = iaaOp(**param_values_dict)
        image_name_suffix = aug_name + '_' + '_'.join(['{}_{}'.format(k, v) for k, v in param_values_dict.items()])
    return aug, image_name_suffix


def generate_all_variants():
    with open("generation_config.yml", 'r') as f:
        config = yaml.full_load(f)
    augmenters = config['augmenters']
    image_names = config['images']

    images = [cv2.imread(image_name) for image_name in image_names]

    for op in augmenters:
        if not os.path.exists(op):
            os.makedirs(op)
        else:
            continue
        augs, image_name_suffixes = parse_aug(op, augmenters[op])
        auged_image_names = ['%s/%s_%s.png' % (op, image_name[:image_name.rfind('.')], image_name_suffix) for
                             image_name_suffix in image_name_suffixes for image_name in image_names]
        auged_images = [image for aug in augs for image in aug.augment_images(images)]
        for i, auged_image in enumerate(auged_images):
            cv2.imwrite(auged_image_names[i], auged_image)


def generate_sample():
    with open("sample_config.yml", 'r') as f:
        config = yaml.load(f)
    sample = config['sample']
    input_folder = sample['input_folder']
    output_folder = sample['output_folder']
    if not os.path.exists(output_folder):
        print('No')
        os.makedirs(output_folder)
    else:
        print('yes')
        #return
    image_names = []
    images = []
    for image_name in glob.glob('%s/*.jpg' % input_folder):
        image_names.append(image_name[image_name.rfind('/') + 1:])
        images.append(cv2.imread(image_name))
    augmenters = sample['augmenters']
    random_list = sample['random']
    sampled_augmenters = [random.sample(augmenters.keys(), r) for r in random_list]
    for i, sampled_augmenter in enumerate(sampled_augmenters):
        sampled_aug = []
        sampled_image_name = []
        for op in sampled_augmenter:
            aug, image_name_suffix = random_aug(op, augmenters[op])
            sampled_aug.append(aug)
            # sampled_image_name.append(image_name_suffix)
            sampled_image_name.append(image_name_suffix[:image_name_suffix.find('_')])
        result_aug = iaa.Sequential(sampled_aug)
        result_image_name = [
            '%s/%s_%s_%s.jpg' % (
                output_folder, image_name[:image_name.rfind('.')], random_list[i], '_'.join(sampled_image_name))
            for image_name in image_names]
        result_auged_image = result_aug.augment_images(images)
        print(result_image_name)
        for j, auged_image in enumerate(result_auged_image):
            cv2.imwrite(result_image_name[j], auged_image)


def batchRandomTransform():
    with open("sample_config.yml", 'r') as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)
    sample = config['sample']
    input_folder = sample['input_folder']
    output_folder = sample['output_folder']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        print('yes')
        #return
    image_names = []
    images = []

    for image_name in glob.glob('%s/*.jpg' % input_folder):
        #print(image_name)
        image_names.append(image_name[image_name.rfind('/') + 1:])
        images.append(cv2.imread(image_name))
    #print(image_names)
    augmenters = sample['augmenters']
    random_list = sample['random']
    random_param = True if sample['random_param'] == 'true' else False
    random_list.sort()
    all_sampled_augmenter = random.sample(augmenters.keys(), random_list[-1])
    sampled_augs = []
    sampled_image_names = []
    prev_i = 0
    for i in random_list:
        for op in all_sampled_augmenter[prev_i:i]:
            aug, image_name_suffix = random_aug(op, augmenters[op])
            sampled_augs.append(aug)
            sampled_image_names.append(image_name_suffix[:image_name_suffix.find('_')])
        result_aug = iaa.Sequential(sampled_augs)
        result_image_name = [
            '%s/%s_%s_%s.jpg' % (
                output_folder, image_name[:image_name.rfind('.')], i, '_'.join(sampled_image_names))
            for image_name in image_names]
        result_auged_image = result_aug.augment_images(images)
        for j, auged_image in enumerate(result_auged_image):
            cv2.imwrite(result_image_name[j], auged_image)
        prev_i = i
        if random_param:
            prev_i = 0
            sampled_augs = []
            sampled_image_names = []


if __name__ == "__main__":
    batchRandomTransform()
