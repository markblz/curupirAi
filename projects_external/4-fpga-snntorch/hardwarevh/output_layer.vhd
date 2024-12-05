library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity output_layer is
    generic (
        NUM_INPUTS  : integer := 128;  -- Ajuste conforme necessário
        NUM_NEURONS : integer := 10;   -- Ajuste conforme necessário
        DATA_WIDTH  : integer := 16;
        BETA        : integer := 15;
        THRESHOLD   : integer := 16
    );
    port (
        clk         : in std_logic;
        reset       : in std_logic;
        input_spike : in std_logic_vector(NUM_INPUTS-1 downto 0);
        spikes_out  : out std_logic_vector(NUM_NEURONS-1 downto 0)
    );
end entity output_layer;

architecture Behavioral of output_layer is

    -- Declaração dos sinais
    type mem_array is array (0 to NUM_NEURONS-1) of integer range -32768 to 32767;
    signal mem_potential : mem_array := (others => 0);
    signal total_current : mem_array := (others => 0);
    signal neuron_spikes : std_logic_vector(NUM_NEURONS-1 downto 0);

    -- Matriz de pesos
    type weight_matrix is array (0 to NUM_NEURONS-1, 0 to NUM_INPUTS-1) of integer;
    constant WEIGHTS : weight_matrix := (
        ( -1, 0, -2, 1, 0, 1, 1, 0, 1, 2, -1, 0, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -2, 1, 1, -1, -1, 0, 0, -1, 0, 0, -1, 0, 1, 1, 0, -1, -1, -1, -1, -2, 1, -1, 0, 0, -1, 0, 0, -1, 0, 1, 1, -2, -1, -1, 1, -1, 2, -1, -1, 1, 1, -1, 1, 0, -1, 1, -1, 0, -1, 0, 0, 1, -1, 1, 0, 0, 0, -1, 1, -2, 1, 1, -1, 0, 1, 1, 0, 1, 0, 1, 0, -1, 0, -1, -1, 1, -2, 0 ),
        ( 0, 0, 0, 0, -1, -1, 0, 0, 1, 0, 0, -1, 2, 0, 1, 0, 0, -1, -1, -2, 1, 0, 0, -1, -2, 2, -1, 0, 0, -1, 0, -2, -1, -1, 1, -1, -1, 2, 0, 0, -1, -1, 0, 1, 2, -1, 0, 0, 0, 0, 2, 1, -2, 0, 1, -1, 1, 2, 0, -1, 0, -1, 0, -1, 0, 1, 0, 1, -1, 0, -1, 1, -1, -1, -1, -2, 1, -2, -1, -1, -1, 0, -1, 0, 1, -1, 0, -1, -1, 1, -1, -1, -1, -1, -1, 1, 2, -1, 0, 1 ),
        ( 0, 1, -1, 0, 0, -1, 0, -1, 0, 0, 1, 1, 1, 1, 1, -1, 0, 1, 1, 0, 1, 1, -1, 1, 1, 0, 0, -2, -1, 0, 1, -1, -1, -1, -1, 0, 1, 2, -1, -1, -1, 2, -1, 0, 2, 0, 1, 0, -1, 1, 1, -1, -2, 0, -1, -1, -1, 0, 0, -2, -1, 1, 1, -1, 1, -1, 0, 1, 1, 1, -1, -1, 0, 1, -1, -1, 1, -1, -1, 1, 1, 0, 1, 1, 1, 0, 1, 1, -1, 0, 0, 1, 1, 1, -2, -1, 0, 0, 1, 1 ),
        ( 1, -1, 0, 0, 0, 0, 1, -1, 0, -1, 0, 1, 1, 0, -1, -1, 0, -1, 0, -1, 0, 1, 1, -1, 0, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 0, -1, 1, 1, 1, 1, 1, -1, 1, 0, 1, 1, -2, 1, -1, 0, 1, 1, -1, -1, -1, -2, -1, 1, 1, -1, 2, 1, 1, -2, 1, 1, -1, 1, -1, 0, -1, 0, -1, -1, 2, -1, 1, 1, -1, 0, 0, 0, -1, 0, 1, -1, 1, -1, 0, 0, -1, 0, 1, 0, -1, 0, 1, -2, 1 ),
        ( 0, -1, -1, -1, 0, -1, -2, -1, 1, 1, -1, -1, -1, 1, 0, 2, -1, 1, -1, 2, 1, -2, -2, 0, 0, 0, 0, 1, -1, -1, 0, 1, 1, -1, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 2, 1, -1, 0, 0, 1, -1, 0, -2, 0, -1, 1, 0, -1, -1, -1, 0, -1, -1, 1, -2, -1, -1, 0, -1, 1, 1, 1, -2, 1, 0, 1, 1, -1, -1, 0, 1, -1, -1, -1, 0, 0, -1, 0, 2, 0, 0, 0, 0, 1, -1, 0, -1, -1 ),
        ( 1, 0, 1, 1, -1, 1, -1, -1, 1, 1, 1, 0, -1, 1, -1, -1, -1, 0, 1, 1, 0, 2, 2, 1, 0, -1, -2, 0, 1, 1, 1, -1, 0, 0, -1, 0, -2, 1, 2, 1, 0, -1, 0, 0, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 0, -1, -1, -1, 1, 0, 1, 0, 0, 0, 0, 1, 2, 1, -1, 0, 0, -1, -2, 1, 1, 1, -1, -1, 1, 0, 1, 1, 0, 0, -1, 0, 0, 0, -1, 0, -1, -1, 1, -2, 0 ),
        ( 2, -2, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, -2, 1, 0, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 0, -1, 0, -1, 0, 1, 0, 0, 0, 2, 1, -2, 1, 1, 1, -1, -1, 0, 0, -1, -1, 1, 1, -1, 0, 1, 0, 0, -1, 0, 0, 0, 0, 1, 0, -1, 1, 0, 0, -1, 1, -1, 1, 1, 0, 0, 1, -1, 1, 0, -1, 1, -1, -1, 2, 0, -1, 0, 0, -1, 1, 1, 1, 0, 1, 1, 0, 1, -2, 1, 1, 0, -1, 0, 1 ),
        ( 0, 0, 0, 2, 0, 0, 1, -1, -1, 1, 2, -1, -1, -1, -2, 1, 0, -1, -1, -1, 0, 0, 1, -2, -1, 1, 0, -2, 1, 0, -1, 2, -2, -1, -1, -1, 0, -1, -1, 0, -1, 0, -2, -1, 0, 0, -1, 0, 0, -1, -1, 1, 0, -1, 2, -1, -1, 1, 1, -2, 0, -1, 1, 0, 0, 0, 1, -1, 0, 1, -1, 1, 2, 1, -1, 1, 1, 0, 0, 1, 0, 0, -1, 1, -1, 2, -2, -1, 0, -1, -1, 0, -1, -1, -1, -1, 0, 0, 0, -1 ),
        ( 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, 1, 1, 0, 1, 1, -1, 1, 1, -1, -1, 1, -1, 0, 1, 0, 0, -1, 1, -1, -2, 1, 1, 0, 1, 0, 0, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 0, 1, 0, -1, -1, -1, 2, -1, -1, -1, 1, 0, 1, -1, 0, 2, 1, -1, -1, -1, 1, 0, -1, 0, -2, 1, 0, 0, -1, 1, 0, 1, -1, 0, 1, 1, -1, 0, 1, 1, 1, -1, 0, -1, 1, -1, 0, 1, 1, -1, -1 ),
        ( -1, 1, -1, 0, 0, -1, 0, -2, 0, -1, 0, -1, 0, -1, 0, 1, 1, -1, 0, -1, 1, -1, -1, 1, 0, 0, 0, 0, 0, 1, 1, -1, 0, -1, 1, 1, 0, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, 1, 2, -1, 0, -1, -1, 1, 0, -1, -1, 0, -1, 0, 1, -1, 0, -1, 0, 1, 0, -1, 2, 1, 1, 0, -1, -1, 0, 1, 2, 1, 0, 1, 1, 0, 0, -1, 0, 2, 0, -1, 1, 1, 1, 0, 0, -1, 0, -1, 0, 1, 0 )
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
