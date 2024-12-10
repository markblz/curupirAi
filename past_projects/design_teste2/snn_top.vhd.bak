library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity snn_top is
    generic (
        NUM_INPUTS    : integer := 784;
        NUM_HIDDEN    : integer := 100;
        NUM_OUTPUTS   : integer := 10;
        NUM_TIME_STEPS: integer := 10
    );
    port (
        clk          : in std_logic;
        reset        : in std_logic;
        input_image  : in std_logic_vector(NUM_INPUTS-1 downto 0);  -- Dados de entrada (pixels)
        output_class : out std_logic_vector(3 downto 0)             -- Classe prevista (0-9)
    );
end entity snn_top;

architecture Behavioral of snn_top is

    -- Sinais para os spikes das camadas
    signal spikes_hidden : std_logic_vector(NUM_HIDDEN-1 downto 0);
    signal spikes_output : std_logic_vector(NUM_OUTPUTS-1 downto 0);

    -- Registradores para acumular os spikes
    type spike_count_array is array (0 to NUM_OUTPUTS-1) of integer range 0 to NUM_TIME_STEPS;
    signal spike_counts : spike_count_array := (others => 0);

    -- Contador de tempo
    signal time_step : integer range 0 to NUM_TIME_STEPS := 0;

begin

    -- Instanciar a camada oculta
    hidden_layer_inst: entity work.hidden_layer
        generic map (
            NUM_INPUTS  => NUM_INPUTS,
            NUM_NEURONS => NUM_HIDDEN,
            DATA_WIDTH  => 16,
            BETA        => 15,
            THRESHOLD   => 16
        )
        port map (
            clk         => clk,
            reset       => reset,
            input_spike => input_image,  -- Neste caso, assumimos que o input_image já é um vetor de spikes
            spikes_out  => spikes_hidden
        );

    -- Instanciar a camada de saída
    output_layer_inst: entity work.output_layer
        generic map (
            NUM_INPUTS  => NUM_HIDDEN,
            NUM_NEURONS => NUM_OUTPUTS,
            DATA_WIDTH  => 16,
            BETA        => 15,
            THRESHOLD   => 16
        )
        port map (
            clk         => clk,
            reset       => reset,
            input_spike => spikes_hidden,
            spikes_out  => spikes_output
        );

    -- Processo para acumular os spikes e determinar a classe
    process(clk, reset)
        variable max_count : integer := 0;
        variable max_index : integer := 0;
    begin
        if reset = '1' then
            spike_counts <= (others => 0);
            time_step <= 0;
            output_class <= (others => '0');
        elsif rising_edge(clk) then
            if time_step < NUM_TIME_STEPS then
                time_step <= time_step + 1;
                -- Acumular os spikes da camada de saída
                for i in 0 to NUM_OUTPUTS-1 loop
                    if spikes_output(i) = '1' then
                        spike_counts(i) <= spike_counts(i) + 1;
                    end if;
                end loop;
            else
                -- Determinar a classe com o maior número de spikes
                max_count := 0;
                max_index := 0;
                for i in 0 to NUM_OUTPUTS-1 loop
                    if spike_counts(i) > max_count then
                        max_count := spike_counts(i);
                        max_index := i;
                    end if;
                end loop;
                output_class <= std_logic_vector(to_unsigned(max_index, 4));
                time_step <= 0;  -- Reiniciar para a próxima inferência
                spike_counts <= (others => 0);
            end if;
        end if;
    end process;

end architecture Behavioral;
