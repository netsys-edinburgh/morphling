ID=$1
FLOPS=$2
MEMORY=$3
UL_BW=$4
DL_BW=$5
UL_LAT=$6
DL_LAT=$7
BACKEND=$8

morphling_device --id $ID --flops $FLOPS --memory $MEMORY --ul_bw $UL_BW --dl_bw $DL_BW --ul_lat $UL_LAT --dl_lat $DL_LAT --backend $BACKEND --cfg /app/config/proxy/cli.ini &
