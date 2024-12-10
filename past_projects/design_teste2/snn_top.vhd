library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity snn_top is
    generic (
        NUM_INPUTS    : integer := 196;  -- Reduzido de 784 para 196
        NUM_HIDDEN    : integer := 25;   -- Reduzido de 100 para 25
        NUM_OUTPUTS   : integer := 10;
        NUM_TIME_STEPS: integer := 5     -- Reduzido de 10 para 5
    );
    port (
        clk          : in std_logic;
        reset        : in std_logic;
        input_image  : in std_logic_vector(NUM_INPUTS-1 downto 0);  -- Tamanho de entrada reduzido
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

    -- Instanciação da camada oculta
    hidden_layer_inst: entity work.hidden_layer
        generic map (
            NUM_INPUTS  => NUM_INPUTS,
            NUM_NEURONS => NUM_HIDDEN,
            DATA_WIDTH  => 8,    -- Reduzido de 16 para 8 bits
            BETA        => 7,    -- Ajustado de acordo com DATA_WIDTH
            THRESHOLD   => 8     -- Ajustado de acordo com DATA_WIDTH
        )
        port map (
            clk         => clk,
            reset       => reset,
            input_spike => input_image,
            spikes_out  => spikes_hidden
        );

    -- Instanciação da camada de saída
    output_layer_inst: entity work.output_layer
        generic map (
            NUM_INPUTS  => NUM_HIDDEN,
            NUM_NEURONS => NUM_OUTPUTS,
            DATA_WIDTH  => 8,    -- Reduzido de 16 para 8 bits
            BETA        => 7,    -- Ajustado de acordo com DATA_WIDTH
            THRESHOLD   => 8     -- Ajustado de acordo com DATA_WIDTH
        )
        port map (
            clk         => clk,
            reset       => reset,
            input_spike => spikes_hidden,
            spikes_out  => spikes_outplibrary IEEE;
            use IEEE.STD_LOGIC_1164.ALL;
            use IEEE.FLOAT_pkg.ALL;
            
            type FloatArray is array (natural range <>) of IEEE.FLOAT32
            
            entity DynamicInputs is
                generic (
                    N : integer := 40  -- Number of inputs, default is 40
                );
                port (
                    clk : in STD_LOGIC;
                    inputs : in STD_LOGIC_VECTOR(N-1 downto 0); 
                    weights: in FloatArray(0 to N-1);
                    output : out STD_LOGIC 
                );
            end entity DynamicInputs;
            
            architecture Behavioral of DynamicInputs is
                signal neuron_input       : FLOAT32 := TO_FLOAT(0.0, FLOAT32);
                signal membrane_potential : FLOAT32 := TO_FLOAT(0.0, FLOAT32);
            begin
                process(clk)
                begin
                    if rising_edge(clk) then
                        for i in 0 to N-1 loop
                            if inputs(i) = '1' then
                                neuron_input := neuron_input + weights(i);
                            end if;
                        end loop;
                        membrane_potential := membrane_potential + TO_FLOAT(0.004, FLOAT32) * (-membrane_potential + 5 * neuron_input)
            
                        if membrane_potential > TO_FLOAT(100.0, FLOAT32) then
                            membrane_potential := TO_FLOAT(0.0, FLOAT32)
                            output <= '1'
                        else
                            output <= '0'
                        end if;
                        
                    end if;
                end process;
            end architecture Behavioral;ut
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
