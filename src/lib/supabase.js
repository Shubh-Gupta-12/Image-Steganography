import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://ltcalqxlhfkidgohhfvh.supabase.co';
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx0Y2FscXhsaGZraWRnb2hoZnZoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzAzODA1NzIsImV4cCI6MjA4NTk1NjU3Mn0.ApV9AWAaTnYkQDsyi3v_8wcOPfv7i_P0Rxn2TyMwsCA';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Storage bucket name
export const STORAGE_BUCKET = 'Stego-Image';
