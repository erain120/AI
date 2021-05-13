def Test_eval(a_model,data_loader,a_device,a_criterion,a_path,a_num_labels):
    print('Start test..')
    checkpoint = torch.load(a_path)
    a_model.load_state_dict(checkpoint)
    a_model.eval()
    correct = 0
    total = 0
    loss = 0
    t_num_label = np.zeros(shape=a_num_labels)
    num_label = np.zeros(shape=a_num_labels)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(a_device), targets.to(a_device)
            total += targets.size(0)

            outputs = model(inputs)

            loss += a_criterion(outputs, targets).item()

            # _, predicted = outputs.max(1)

            _, predicted = torch.max(outputs,1)
            # correct += predicted.eq(targets).sum().item()
            correct += (targets == predicted).sum().item()
            t_num_label[targets] += 1
            num_label[predicted] += (targets == predicted).sum().item()

        y_graph = []
        print('\nTotal average test acc : ', correct / total)
        print('total average test_loss :', loss / total)

        for i in range(a_num_labels):
            print('{} label Accuracy of {:.2f} '.format(i, num_label[i]/t_num_label[i]*100))
            y_graph.append(num_label[i]/t_num_label[i]*100)

        x = np.arange(a_num_labels)
        label_name = np.arange(a_num_labels)
        plt.figure(figsize=(10, 8))
        plt.bar(x, y_graph, width=0.6, align="center")
        plt.xticks(x, label_name, fontsize="10", rotation=45)
        plt.show()