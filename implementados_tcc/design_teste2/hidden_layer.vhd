library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity hidden_layer is
    generic (
        NUM_INPUTS  : integer := 196;   -- Ajustado para corresponder ao novo NUM_INPUTS
        NUM_NEURONS : integer := 25;    -- Ajustado para corresponder ao novo NUM_HIDDEN
        DATA_WIDTH  : integer := 8;     -- Reduzido para 8 bits
        BETA        : integer := 240;   -- Aproximadamente 0.9375 * 256
        THRESHOLD   : integer := 128    -- Limiar de disparo ajustado
    );
    port (
        clk         : in std_logic;
        reset       : in std_logic;
        input_spike : in std_logic_vector(NUM_INPUTS-1 downto 0);
        spikes_out  : out std_logic_vector(NUM_NEURONS-1 downto 0)
    );
end entity hidden_layer;

architecture Behavioral of hidden_layer is

    -- Declaração dos sinais
    type mem_array is array (0 to NUM_NEURONS-1) of integer range -128 to 127;
    signal mem_potential : mem_array := (others => 0);
    signal total_current : mem_array := (others => 0);
    signal neuron_spikes : std_logic_vector(NUM_NEURONS-1 downto 0);

    -- Matriz de pesos ajustada
    type weight_matrix is array (0 to NUM_NEURONS-1, 0 to NUM_INPUTS-1) of integer range -128 to 127;
    constant WEIGHTS : weight_matrix := (
        -- Inserir aqui os novos pesos ajustados (25x196)
        -- Exemplo:
        ( 1, -1, 0, ..., 1 ),  -- Neurônio 0
        ( 0, 1, -1, ..., 0 ),  -- Neurônio 1
        -- ...
        ( -1, 0, 1, ..., -1 )  -- Neurônio 24
    );

begin

    -- Processo para calcular as correntes totais
    calc_total_current: process(input_spike)
        variable temp_total_current : mem_array;
    begin
        for i in 0 to NUM_NEURONS-1 loop
            temp_total_current(i) := 0;
            for j in 0 to NUM_INPUTS-1 loop
                if input_spike(j) = '1' then
                    temp_total_current(i) := temp_total_current(i) + WEIGHTS(i, j);
                end if;
            end loop;
        end loop;
        total_current <= temp_total_current;
    end process;

    -- Instanciação dos neurônios
    gen_neurons: for i in 0 to NUM_NEURONS-1 generate
        neuron_inst: entity work.neuron_lif
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            BETA       => BETA,
            THRESHOLD  => THRESHOLD
        )
        port map (
            clk         => clk,
            reset       => reset,
            input_I     => total_current(i),
            mem_pot_out => mem_potential(i),
            spike_out   => neuron_spikes(i)
        );
    end generate;

    -- Atribuição das saídas
    spikes_out <= neuron_spikes;

end architecture Behavioral;
