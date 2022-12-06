from common.eval import *

model.eval()

if P.mode == 'test_acc':
    from evals import test_classifier
    with torch.no_grad():
        error = test_classifier(P, model, test_loader, 0, logger=None)

elif P.mode == 'test_marginalized_acc':
    from evals import test_classifier
    with torch.no_grad():
        error = test_classifier(P, model, test_loader, 0, marginal=True, logger=None)

elif P.mode in ['ood', 'ood_pre']:
    if P.mode == 'ood':
        from evals import eval_ood_detection
    else:
        from evals.ood_pre import eval_ood_detection

    with torch.no_grad():
        auroc_dict, report_dict, logit_auroc_dict = eval_ood_detection(P, model, test_loader, ood_test_loader, P.ood_score,
                                        train_loader=train_loader, simclr_aug=simclr_aug)

    if P.one_class_idx is not None:
        mean_dict = dict()
        for ood_score in P.ood_score:
            mean = 0
            for ood in auroc_dict.keys():
                mean += auroc_dict[ood][ood_score]
            mean_dict[ood_score] = mean / len(auroc_dict.keys())
        auroc_dict['one_class_mean'] = mean_dict

    bests = []
    for ood in auroc_dict.keys():
        message = ''
        best_auroc = 0
        for ood_score, auroc in auroc_dict[ood].items():
            message += '[%s %s %.4f] ' % (ood, ood_score, auroc)
            if auroc > best_auroc:
                best_auroc = auroc
        message += '[%s %s %.4f] ' % (ood, 'best', best_auroc)
        if P.print_score:
            print(message)
        bests.append(best_auroc)

    bests = map('{:.4f}'.format, bests)
    print('\t'.join(bests))

    if P.dataset in ['bollworms', 'bollworms-clean']:
        print('')
        print('-'*80)
        print('')
        for ood in report_dict.keys():
            for ood_score, (class_report, conf_matrix) in report_dict[ood].items():

                print(f'F1-optimized classification report [OOD: {ood}, Score: {ood_score}]:')
                print('')
                print(class_report)

                per_class_accuracy = conf_matrix.diagonal()/conf_matrix.sum(axis=1)
                print(f'OOD accuracy: {per_class_accuracy[0]:.3f}')
                print(f' ID accuracy: {per_class_accuracy[1]:.3f}')
                print('')

                tn, fp, fn, tp = conf_matrix.ravel()
                print('Breakdown:', {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})
                print('')

                print(f'AUC: {logit_auroc_dict[ood][ood_score]:.3f}')
                print('')

                print('-'*80)
                print('')

else:
    raise NotImplementedError()


