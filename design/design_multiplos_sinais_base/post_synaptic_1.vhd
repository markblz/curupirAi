library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.FLOAT_pkg.ALL;

entity post_synaptic_1 is
    generic (
        N : integer := 40  -- Number of inputs, default is 40
    );
    port (
        clk : in STD_LOGIC;
        reset : in STD_LOGIC;
        inputs : in STD_LOGIC_VECTOR(N-1 downto 0); 
        output : out STD_LOGIC 
    );
end entity post_synaptic_1;

architecture Behavioral_1 of post_synaptic_1 is
    signal neuron_input       : integer := 0;
    signal membrane_potential : integer := 0;
    type mem_array is array (0 to N - 1) of integer;
    constant weights : mem_array := (
        1581, 2467, 2211,   25,   25,   25,   25,   25,   25,   25,   25, 25,   25,   25,   25,  732, 1866, 2408,  984, 1278, 2321, 6205, 25,   25,   25,   25,   25,   25,   25,   25, 5581, 5045,25, 25,   25,   25,   25,   25,   25,   25
    );
begin
    process(inputs)
    begin
            for i in 0 to N-1 loop
                if inputs(i) = '1' then
                    neuron_input <= neuron_input + weights(i);
                end if;
            end loop;
            membrane_potential <= membrane_potential +  neuron_input;

            if membrane_potential > 4000 then
                membrane_potential <= 0;
                output <= '1';
            else
                output <= '0';
            end if;
    end process;

    process(reset)
    begin
        if reset = '1' then
            neuron_input <= 0;
            membrane_potential <= 0;
        end if;
    end process;

end architecture Behavioral_1;