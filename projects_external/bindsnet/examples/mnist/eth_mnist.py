import argparse  # Importa a biblioteca para analise de argumentos da linha de comando
import os  # Importa a biblioteca para interacoes com o sistema operacional
from time import time as t  # Importa a funcao time e a renomeia para t

import matplotlib.pyplot as plt  # Importa a biblioteca para plotagem de graficos
import numpy as np  # Importa a biblioteca para operacoes numericas
import torch  # Importa a biblioteca para operacoes com tensores e aprendizado de maquina
from torchvision import transforms  # Importa a biblioteca para transformacoes de dados de visao computacional
from tqdm import tqdm  # Importa a biblioteca para criar barras de progresso

# Importa funcoes e classes especificas do bindsnet
from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights

# Configura o parser de argumentos da linha de comando
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)  # Semente para geracao de numeros aleatorios
parser.add_argument("--n_neurons", type=int, default=100)  # Numero de neuronios
parser.add_argument("--n_epochs", type=int, default=1)  # Numero de epocas de treinamento
parser.add_argument("--n_test", type=int, default=10000)  # Numero de amostras de teste
parser.add_argument("--n_train", type=int, default=60000)  # Numero de amostras de treinamento
parser.add_argument("--n_workers", type=int, default=-1)  # Numero de trabalhadores para carregamento de dados
parser.add_argument("--exc", type=float, default=22.5)  # Excitacao
parser.add_argument("--inh", type=float, default=120)  # Inibicao
parser.add_argument("--theta_plus", type=float, default=0.05)  # Incremento de limiar
parser.add_argument("--time", type=int, default=250)  # Tempo de simulacao
parser.add_argument("--dt", type=float, default=1.0)  # Passo de tempo
parser.add_argument("--intensity", type=float, default=128)  # Intensidade do estimulo
parser.add_argument("--progress_interval", type=int, default=10)  # Intervalo de progresso
parser.add_argument("--update_interval", type=int, default=250)  # Intervalo de atualizacao
parser.add_argument("--train", dest="train", action="store_true")  # Modo de treinamento
parser.add_argument("--test", dest="train", action="store_false")  # Modo de teste
parser.add_argument("--plot", dest="plot", action="store_true")  # Habilitar plotagem
parser.add_argument("--gpu", dest="gpu", action="store_true")  # Habilitar uso de GPU
parser.set_defaults(plot=True, gpu=True)  # Define valores padrao para plot e gpu

# Analisa os argumentos da linha de comando
args = parser.parse_args()

# Atribui os argumentos a variaveis
seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu

# Configura o uso da GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)  # Define a semente para todas as GPUs
else:
    torch.manual_seed(seed)  # Define a semente para a CPU
    device = "cpu"
    if gpu:
        gpu = False  # Desabilita o uso de GPU se nao estiver disponivel

torch.set_num_threads(os.cpu_count() - 1)  # Define o numero de threads para o PyTorch
print("Running on Device = ", device)  # Imprime o dispositivo em uso

# Determina o numero de trabalhadores para carregamento de dados
if n_workers == -1:
    n_workers = 0  # Define o numero de trabalhadores como 0

if not train:
    update_interval = n_test  # Atualiza o intervalo de atualizacao para o numero de testes

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))  # Calcula a raiz quadrada do numero de neuronios
start_intensity = intensity  # Define a intensidade inicial

# Constroi a rede neural usando o modelo DiehlAndCook2015
network = DiehlAndCook2015(
    n_inpt=784,  # Numero de entradas
    n_neurons=n_neurons,  # Numero de neuronios
    exc=exc,  # Excitacao
    inh=inh,  # Inibicao
    dt=dt,  # Passo de tempo
    norm=78.4,  # Normalizacao
    theta_plus=theta_plus,  # Incremento de limiar
    inpt_shape=(1, 28, 28),  # Formato da entrada
)

# Direciona a rede para a GPU, se disponivel
if gpu:
    network.to("cuda")

# Carrega os dados de treinamento do MNIST
train_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),  # Codificador Poisson
    None,
    root=os.path.join("..", "..", "data", "MNIST"),  # Caminho para os dados
    download=True,  # Baixa os dados se nao estiverem disponiveis
    train=True,  # Define o modo de treinamento
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]  # Transforma os dados
    ),
)

# Registra os picos durante a simulacao
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Atribuicoes de neuronios e proporcoes de picos
n_classes = 10  # Numero de classes
assignments = -torch.ones(n_neurons, device=device)  # Atribuicoes de neuronios
proportions = torch.zeros((n_neurons, n_classes), device=device)  # Proporcoes de picos
rates = torch.zeros((n_neurons, n_classes), device=device)  # Taxas de picos

# Sequencia de estimativas de precisao
accuracy = {"all": [], "proportion": []}

# Monitoramento de voltagem para camadas excitadoras e inibidoras
exc_voltage_monitor = Monitor(
    network.layers["Ae"], ["v"], time=int(time / dt), device=device
)
inh_voltage_monitor = Monitor(
    network.layers["Ai"], ["v"], time=int(time / dt), device=device
)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Configura monitores para picos e voltagens
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

# Variaveis para plotagem
inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

# Treina a rede neural
print("\nBegin training.\n")
start = t()
for epoch in range(n_epochs):
    labels = []

    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Cria um dataloader para iterar e agrupar dados
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(dataloader)):
        if step > n_train:
            break
        # Obtem a proxima amostra de entrada
        inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        if step % update_interval == 0 and step > 0:
            # Converte o array de rotulos em um tensor
            label_tensor = torch.tensor(labels, device=device)

            # Obtem as previsoes da rede
            all_activity_pred = all_activity(
                spikes=spike_record, assignments=assignments, n_labels=n_classes
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )

            # Calcula a precisao da rede de acordo com as estrategias de classificacao disponiveis
            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
            )
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred).item()
                / len(label_tensor)
            )

            print(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            print(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                " (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )

            # Atribui rotulos aos neuronios da camada excitadora
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )

            labels = []

        labels.append(batch["label"])

        # Executa a rede na entrada
        network.run(inputs=inputs, time=time)

        # Obtem o registro de voltagem
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        # Adiciona ao registro de picos
        spike_record[step % update_interval] = spikes["Ae"].get("s").squeeze()

        # Opcionalmente, plota varias informacoes da simulacao
        if plot:
            image = batch["image"].view(28, 28)
            inpt = inputs["X"].view(time, 784).sum(0).view(28, 28)
            input_exc_weights = network.connections[("X", "Ae")].w
            square_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )
            square_assignments = get_square_assignments(assignments, n_sqrt)
            spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
            voltages = {"Ae": exc_voltages, "Ai": inh_voltages}
            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=batch["label"], axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(accuracy, x_scale=update_interval, ax=perf_ax)
            voltage_ims, voltage_axes = plot_voltages(
                voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            )

            plt.pause(1e-8)

        network.reset_state_variables()  # Reseta as variaveis de estado

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")

# Carrega os dados de teste do MNIST
test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Sequencia de estimativas de precisao
accuracy = {"all": 0, "proportion": 0}

# Registra os picos durante a simulacao
spike_record = torch.zeros((1, int(time / dt), n_neurons), device=device)

# Testa a rede neural
print("\nBegin testing\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)
for step, batch in enumerate(test_dataset):
    if step >= n_test:
        break
    # Obtem a proxima amostra de entrada
    inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Executa a rede na entrada
    network.run(inputs=inputs, time=time)

    # Adiciona ao registro de picos
    spike_record[0] = spikes["Ae"].get("s").squeeze()

    # Converte o array de rotulos em um tensor
    label_tensor = torch.tensor(batch["label"], device=device)

    # Obtem as previsoes da rede
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Calcula a precisao da rede de acordo com as estrategias de classificacao disponiveis
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reseta as variaveis de estado
    pbar.set_description_str("Test progress: ")
    pbar.update()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")